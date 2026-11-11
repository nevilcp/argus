"""Smoke tests for the CLI entry points under scripts/.

Each script gets a `--help` check (argument parsing wires up and exits 0), and
`collect_session`, `reconcile_outcomes`, `run_evaluation` and `train_macro_hmm`
also get one `main()` run against fixture-backed or injected dependencies. The
`main()` runs go through the real shared code paths; only the network edges
(the MFT pipeline sweep, market data) are replaced.
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from argus.agents.macro import MacroStatisticalAgent
from argus.agents.macro_hmm import RegimeClassifier
from argus.config import settings
from scripts import collect_session, reconcile_outcomes, run_evaluation, train_macro_hmm
from tests.helpers.builders import argus_decision, neutral_cultural_memory
from tests.helpers.market_data import FakeMarketData
from tests.helpers.paths import FIXTURES_DIR

_SCRIPTS = [
    "collect_session",
    "reconcile_outcomes",
    "replay_backtest",
    "run_evaluation",
    "train_macro_hmm",
]


@pytest.mark.parametrize("name", _SCRIPTS)
def test_help_prints_usage_and_exits_zero(name, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        importlib.import_module(f"scripts.{name}").main()

    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_collect_session_rejects_an_unknown_risk_tolerance(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["collect_session", "--risk-tolerance", "YOLO"])

    with pytest.raises(SystemExit) as exc_info:
        collect_session.main()

    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def _run_collect_session(
    monkeypatch, tmp_path: Path, *, session_states: dict, final_state: dict, extra_args=()
) -> mock.Mock:
    """Runs collect_session.main() over a fake pipeline and graph; returns the graph."""
    pipeline = mock.Mock()
    pipeline.run_once = mock.AsyncMock(return_value=session_states)
    graph = mock.Mock()
    graph.invoke.return_value = final_state
    monkeypatch.setattr(collect_session, "MFTDataPipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(collect_session, "build_graph", lambda **kwargs: graph)
    argv = [
        "collect_session",
        "--universe",
        "AAPL, MSFT",
        "--result-out",
        str(tmp_path / "result.json"),
        *extra_args,
    ]
    monkeypatch.setattr("sys.argv", argv)
    collect_session.main()
    return graph


def test_collect_session_main_with_no_session_data_is_a_clean_noop(monkeypatch, tmp_path, capsys):
    graph = _run_collect_session(monkeypatch, tmp_path, session_states={}, final_state={})

    graph.invoke.assert_not_called()
    assert "ran=False" in capsys.readouterr().out
    assert json.loads((tmp_path / "result.json").read_text())["ran"] is False


def test_collect_session_main_exits_1_when_the_cycle_is_degraded(monkeypatch, tmp_path, capsys):
    decision = argus_decision("AAPL", datetime(2024, 1, 2))  # no allocation

    with pytest.raises(SystemExit) as exc_info:
        _run_collect_session(
            monkeypatch,
            tmp_path,
            session_states={"AAPL": {"timestamp": "2024-01-02T09:30:00-05:00"}},
            final_state={"decisions": [decision], "errors": [], "macro_context": None},
        )

    assert exc_info.value.code == 1
    assert "::error::collection cycle DEGRADED" in capsys.readouterr().out
    assert (Path(settings.ARGUS_DATA_DIR) / "decisions.jsonl").exists()
    assert json.loads((tmp_path / "result.json").read_text())["outcome"] == "degraded"


def test_reconcile_outcomes_main_reports_an_empty_log(monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr("sys.argv", ["reconcile_outcomes", "--decisions-log", str(log_path)])
    monkeypatch.setattr(reconcile_outcomes, "get_cultural_memory", neutral_cultural_memory)

    reconcile_outcomes.main(market_data=FakeMarketData())

    assert f"Loaded 0 decision(s) from {log_path}" in capsys.readouterr().out


def test_run_evaluation_main_prints_the_report_for_the_fixture_session(monkeypatch, capsys):
    # The fixture session was captured on 2026-05-22, so these prices cover its forward window.
    prices = pd.DataFrame(
        {"close": [100.0 + i for i in range(30)]},
        index=pd.date_range("2026-05-20", periods=30, freq="D"),
    )
    market_data = FakeMarketData(ohlcv_daily=lambda ticker, period="2y": prices)
    monkeypatch.setattr("sys.argv", ["run_evaluation", "--fixtures-dir", str(FIXTURES_DIR)])
    # The closed-loop replay reads the real cultural memory unless told otherwise.
    monkeypatch.setattr("argus.orchestration.graph.get_cultural_memory", neutral_cultural_memory)

    run_evaluation.main(market_data=market_data)

    out = capsys.readouterr().out
    assert "=== Pre-registered evaluation" in out
    # The fixture session allocates to one ticker, too few for a rank IC.
    assert "--- Open-loop (n=1)" in out
    assert "--- Closed-loop (n=1)" in out
    assert "open-loop replay is deterministic: True" in out


def test_train_macro_hmm_main_exits_1_when_the_fit_fails(monkeypatch, tmp_path, capsys):
    output = tmp_path / "hmm.joblib"
    monkeypatch.setattr("sys.argv", ["train_macro_hmm", "--output", str(output)])

    with pytest.raises(SystemExit) as exc_info:
        train_macro_hmm.main(market_data=FakeMarketData(fred_series=RuntimeError("FRED down")))

    assert exc_info.value.code == 1
    assert "Fit failed" in capsys.readouterr().out
    assert not output.exists()


def test_train_macro_hmm_main_saves_a_validated_classifier(monkeypatch, tmp_path, capsys):
    output = tmp_path / "hmm.joblib"
    monkeypatch.setattr("sys.argv", ["train_macro_hmm", "--output", str(output)])
    # The fit itself is covered in test_macro.py; here the committed artifact
    # stands in for a freshly fit model so main()'s report-validate-save path runs.
    monkeypatch.setattr(MacroStatisticalAgent, "fit_on_history", lambda self, start_date: True)

    train_macro_hmm.main(market_data=FakeMarketData())

    assert "Validation gate passed" in capsys.readouterr().out
    assert RegimeClassifier.load(output).is_fitted
