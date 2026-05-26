"""Agent-system implementations + registry.

Three reference systems are provided:

- ``single_react``: a single-agent ReAct loop.
- ``planner_worker``: a two-agent sequential planner+worker agent system.
- ``langgraph``: a LangGraph-based ReAct wrapper, included as a
  comparison option against a popular off-the-shelf framework.

``AGENT_SYSTEMS`` maps a name string to the implementing class — benchmark
runners resolve a system by name.
"""

from __future__ import annotations

from ._protocol import AgentSystem, AgentSystemResult
from .langgraph import LangGraphSystem
from .planner_worker import PlannerWorkerSystem
from .single_react import SingleReActSystem

AGENT_SYSTEMS: dict[str, type[AgentSystem]] = {
    "single_react": SingleReActSystem,
    "planner_worker": PlannerWorkerSystem,
    # LangGraphSystem.name is a ClassVar[str] which Protocol structural
    # matching treats as distinct from instance `name: str`; runtime
    # conformance holds.
    "langgraph": LangGraphSystem,  # type: ignore[dict-item]
}


__all__ = [
    "AGENT_SYSTEMS",
    "AgentSystem",
    "AgentSystemResult",
    "LangGraphSystem",
    "PlannerWorkerSystem",
    "SingleReActSystem",
]
