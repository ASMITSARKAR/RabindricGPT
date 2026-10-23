"""
tests/preflight/test_stages.py : Tests for preflight stage registry, dependency ordering, and failure propagation.
"""

from src.rgpt.preflight.stages import PreflightRegistry


def test_stages_dependency_order() -> None:
    """Registered checks must execute in valid topological order respecting 'needs' dependencies."""
    registry = PreflightRegistry()
    execution_order = []

    def check_b() -> bool:
        execution_order.append("B")
        return True

    def check_a() -> bool:
        execution_order.append("A")
        return True

    # A needs B, but registered first
    registry.register_check("P_test", "check_A", check_a, needs=["check_B"])
    registry.register_check("P_test", "check_B", check_b, needs=[])

    res = registry.run_level("P_test")
    assert res["status"] == "pass"
    assert execution_order == ["B", "A"]


def test_stages_failure_blocks_level() -> None:
    """Failure in one check must stop execution of its dependent checks and mark level failed."""
    registry = PreflightRegistry()
    executed = []

    def failing_check() -> bool:
        executed.append("failing")
        return False

    def dependent_check() -> bool:
        executed.append("dependent")
        return True

    registry.register_check("P_test", "check_fail", failing_check)
    registry.register_check("P_test", "check_dep", dependent_check, needs=["check_fail"])

    res = registry.run_level("P_test")
    assert res["status"] == "fail"
    assert res["checks"]["check_fail"] == "fail"
    assert res["checks"]["check_dep"] == "blocked"
    assert "dependent" not in executed
