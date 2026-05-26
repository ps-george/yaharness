"""Local re-export of the `AgentSystem` protocol.

Re-exports the canonical types from the benchmark layer so the agent
modules satisfy the benchmark `AgentSystem` protocol by construction.
"""

from __future__ import annotations

from ..benchmarks.outcome import AgentSystemResult
from ..benchmarks.protocol import AgentSystem

__all__ = ["AgentSystem", "AgentSystemResult"]
