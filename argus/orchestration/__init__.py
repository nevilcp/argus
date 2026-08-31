"""Orchestration sub-package for ARGUS.

Holds the shared graph state (state.py), the LangGraph DAG (graph.py), signal
aggregation (aggregator.py), rate-limit governance (governor.py), the unattended
collection cycle (collector.py), and outcome reconciliation (reconciliation.py).
Callers import each module directly; nothing is re-exported here.
"""
