"""Orchestration sub-package for ARGUS.

Holds the shared graph state (state.py), the LangGraph DAG (graph.py) and its node
functions (nodes.py), signal aggregation (aggregator.py), rate-limit governance
(governor.py), the unattended collection cycle (collector.py), the decision log
(decision_log.py), credit assignment (attribution.py), and outcome reconciliation
(reconcile.py).
Callers import each module directly; nothing is re-exported here.
"""
