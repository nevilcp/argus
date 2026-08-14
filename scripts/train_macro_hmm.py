# ruff: noqa: E402
"""
scripts/train_macro_hmm.py

CLI entry point to fit RegimeClassifier (argus/agents/macro.py) on a long FRED +
VIX history and persist the result as the artifact every MacroStatisticalAgent
construction site loads.

    .venv/bin/python -m scripts.train_macro_hmm [--start-date 2010-01-01] [--output PATH]

Offline-only: calls MacroStatisticalAgent.fit_on_history(), which does its own
direct yf.download() for VIX history and fetch_fred_series() calls for FRED
series — both real network calls, not fixture-backed. Not run by CI or any
scheduled workflow; retraining is manual (see the README's Deployment section).
"""

import argparse
import logging

from dotenv import load_dotenv

# .env must be loaded before any LangChain/Groq imports that read env vars
load_dotenv()

from argus.agents.macro import MacroStatisticalAgent
from argus.config import settings
from argus.schemas.signals import Regime

logger = logging.getLogger("argus.train_macro_hmm")


def main() -> None:
    """Parses CLI args, fits the HMM on FRED/VIX history, and saves the artifact."""
    logging.basicConfig(level=settings.ARGUS_LOG_LEVEL)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        default="2010-01-01",
        help="ISO date string for the beginning of the training window",
    )
    parser.add_argument(
        "--output",
        default=settings.ARGUS_HMM_MODEL_PATH,
        help="Destination path for the persisted classifier artifact",
    )
    args = parser.parse_args()

    agent = MacroStatisticalAgent()
    agent.fit_on_history(start_date=args.start_date)

    classifier = agent.classifier
    if not classifier.is_fitted:
        print("Fit failed — classifier.is_fitted is False. Check the error logged above.")
        raise SystemExit(1)

    classifier.save(args.output, start_date=args.start_date)

    print(f"Saved artifact to {args.output} ({classifier.n_train_observations} observations).")
    print("\nPer-state feature means and assigned regime label:")
    for state in sorted(classifier.state_means):
        means = classifier.state_means[state]
        regime = classifier.state_to_regime.get(state, Regime.TRANSITIONAL.value)
        formatted = ", ".join(f"{col}={val:.3f}" for col, val in means.items())
        print(f"  state {state} -> {regime:<12} {formatted}")


if __name__ == "__main__":
    main()
