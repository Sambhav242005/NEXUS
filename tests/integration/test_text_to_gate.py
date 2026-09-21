"""Text -> planner intent -> SemIf gate (mock scorer, real validation)."""
from nexus.semif.adapter import RealSemIfAdapter, SemIfConfig
from nexus.semif.decisions import TOOL_ROUTING
from nexus.safety.policy import check_action


def test_text_to_gate_flow():
    ad = RealSemIfAdapter(SemIfConfig())
    res = ad.decide({"utterance": "search nvidia driver"}, "Which tool?", TOOL_ROUTING, "flow-1")
    assert res.winner_id in [o.id for o in TOOL_ROUTING]
    verdict = check_action(res.winner_id, {"intent": "search"})
    assert verdict in ("allow", "confirm", "deny")
