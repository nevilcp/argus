"""
scripts/train_macro_hmm.py

CLI entry point to fit RegimeClassifier (argus/agents/macro_hmm.py) on a long FRED +
VIX history and persist the result as the artifact every MacroStatisticalAgent
construction site loads.

    .venv/bin/python -m scripts.train_macro_hmm [--start-date 1990-01-01] [--output PATH]

Offline-only: calls MacroStatisticalAgent.fit_on_history(), which fetches VIX
and FRED history through the agent's MarketDataProvider (live by default, so
real network calls). Not run by CI or any scheduled workflow; retraining is
manual (see the README's Deployment section).

The artifact is only written if RegimeClassifier.validation_failures() comes
back empty — a non-zero exit here means the fit produced a model that isn't a
function of the macro data, and no artifact is written for it.
"""

import argparse

from argus.agents.macro import MacroStatisticalAgent
from argus.config import settings
from argus.infra.log import configure_logging
from argus.params import MACRO
from argus.ports import MarketDataProvider
from argus.schemas.signals import Regime


def main(*, market_data: MarketDataProvider | None = None) -> None:
    """Parses CLI args, fits the HMM, and saves the artifact if it validates.

    Args:
        market_data: History source; defaults to live fetches. Tests inject a fixture-backed one.
    """
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        default=MACRO.hmm_training_start,
        help="ISO date string for the beginning of the training window",
    )
    parser.add_argument(
        "--output",
        default=settings.ARGUS_HMM_MODEL_PATH,
        help="Destination path for the persisted classifier artifact",
    )
    args = parser.parse_args()

    agent = MacroStatisticalAgent(market_data=market_data)
    if not agent.fit_on_history(start_date=args.start_date):
        print("Fit failed. Check the error logged above.")
        raise SystemExit(1)

    classifier = agent.classifier

    print("\nPer-state feature means and assigned regime label:")
    for state in sorted(classifier.state_means):
        means = classifier.state_means[state]
        regime = classifier.state_to_regime.get(state, Regime.TRANSITIONAL.value)
        formatted = ", ".join(f"{col}={val:.3f}" for col, val in means.items())
        print(f"  state {state} -> {regime:<12} {formatted}")

    print("\nValidation metrics:")
    for key, value in classifier.validation_metrics.items():
        print(f"  {key}: {value}")

    failures = classifier.validation_failures()
    if failures:
        print("\nValidation gate FAILED — artifact not saved:")
        for reason in failures:
            print(f"  - {reason}")
        raise SystemExit(1)

    classifier.save(args.output, start_date=args.start_date)
    print(
        f"\nValidation gate passed. Saved artifact to {args.output} "
        f"({classifier.n_train_observations} observations)."
    )


if __name__ == "__main__":
    main()
