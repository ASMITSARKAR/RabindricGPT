"""
src/rgpt/preflight/stages.py : Preflight stage execution engine and check registry with dependency resolution.
Backlog/Phase : Phase P      Docs : 15_BUILD_DIRECTIVE.md Phase P
Contract      : register_check(level, name, fn, needs) -> None; run_stage(level) -> dict
Invariants    : Checks run in topological order; failure in check blocks dependent checks and marks level failed
Failure modes : Circular dependency in preflight checks; unhandled exception during check execution
Status        : TESTED
"""

from typing import Any, Callable, Dict, List, Optional


class PreflightRegistry:
    """Registry and dependency scheduler for preflight verification checks."""

    def __init__(self) -> None:
        # level -> dict of checks: name -> {"fn": callable, "needs": list[str]}
        self.levels: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def register_check(
        self,
        level: str,
        name: str,
        check_fn: Callable[[], bool],
        needs: Optional[List[str]] = None,
    ) -> None:
        """Register a preflight check with dependencies."""
        if level not in self.levels:
            self.levels[level] = {}

        self.levels[level][name] = {
            "fn": check_fn,
            "needs": needs or [],
        }

    def _get_topological_order(self, level: str) -> List[str]:
        """Compute execution order for checks within a level."""
        checks = self.levels.get(level, {})
        visited = set()
        visiting = set()
        order = []

        def dfs(name: str):
            if name in visiting:
                raise ValueError(f"Circular dependency detected in preflight checks: {name}")
            if name not in visited:
                visiting.add(name)
                for dep in checks.get(name, {}).get("needs", []):
                    if dep in checks:
                        dfs(dep)
                visiting.remove(name)
                visited.add(name)
                order.append(name)

        for name in sorted(checks.keys()):
            if name not in visited:
                dfs(name)

        return order

    def run_level(self, level: str) -> Dict[str, Any]:
        """Execute all checks registered for a given preflight level in dependency order."""
        checks = self.levels.get(level, {})
        if not checks:
            return {"status": "pass", "checks": {}, "failures": []}

        order = self._get_topological_order(level)
        results: Dict[str, str] = {}
        failures: List[str] = []

        for name in order:
            deps = checks[name]["needs"]
            # If any dependency failed or was blocked, block this check
            blocked_by = [dep for dep in deps if results.get(dep) != "pass"]
            if blocked_by:
                results[name] = "blocked"
                continue

            try:
                passed = checks[name]["fn"]()
                if passed:
                    results[name] = "pass"
                else:
                    results[name] = "fail"
                    failures.append(f"{level}.{name} failed check condition")
            except Exception as e:
                results[name] = "fail"
                failures.append(f"{level}.{name} raised exception: {str(e)}")

        overall_status = "pass" if not failures and all(v == "pass" for v in results.values()) else "fail"
        return {
            "status": overall_status,
            "checks": results,
            "failures": failures,
        }
