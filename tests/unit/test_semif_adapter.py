"""Real SemIf adapter tests — use real validation path, mock scorer (no GPU)."""
import pytest
from nexus.semif.adapter import RealSemIfAdapter, SemIfConfig, SEMIF_AVAILABLE
from nexus.semif.schemas import DecisionOption
from nexus.semif.decisions import TOOL_ROUTING, ACTION_GATING

OPTS = [DecisionOption(id="a", description="Option A."), DecisionOption(id="b", description="Option B.")]


def test_real_validation_rejects_malformed():
    if not SEMIF_AVAILABLE:
        pytest.skip("real SemIf core unavailable")
    ad = RealSemIfAdapter(SemIfConfig())
    with pytest.raises(ValueError):
        ad.decide("", "Q?", OPTS)  # empty state rejected by real validate_row
    with pytest.raises(ValueError):
        ad.decide("s", "Q?", [DecisionOption(id="only", description="One.")])  # <2 options


def test_decide_routes_and_gates():
    if not SEMIF_AVAILABLE:
        pytest.skip("real SemIf core unavailable")
    ad = RealSemIfAdapter(SemIfConfig())
    r = ad.decide({"utterance": "open chrome"}, "Which tool?", TOOL_ROUTING, "t1")
    assert r.winner_id in [o.id for o in TOOL_ROUTING]
    assert abs(sum(r.probabilities) - 1.0) < 1e-6
    assert r.probability_status.startswith("conditional")
    g = ad.decide({"action": "click x=10 y=20"}, "Gate?", ACTION_GATING, "g1")
    assert g.winner_id in ("allow", "ask_confirmation", "reject")


def test_timeout_enforced():
    import time as _t
    if not SEMIF_AVAILABLE:
        pytest.skip("real SemIf core unavailable")

    def slow(row):
        _t.sleep(0.05)
        return {"option_ids": ["a", "b"], "probabilities": [0.5, 0.5]}
    ad = RealSemIfAdapter(SemIfConfig(timeout_seconds=0.001), scorer=slow)
    with pytest.raises(TimeoutError):
        ad.decide("state", "Q?", OPTS)
