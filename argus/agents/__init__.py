"""Package marker for the six specialist agents.

Each agent lives in its own module (technical, macro, fundamental, sentiment,
risk, portfolio) and is imported from that module directly. This package
deliberately re-exports nothing, so importing one agent never drags in the
other five.
"""
