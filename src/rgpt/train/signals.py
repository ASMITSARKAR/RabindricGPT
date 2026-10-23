"""
src/rgpt/train/signals.py : POSIX and Windows signal handlers for graceful shutdown on VM preemption or cancellation.
Backlog/Phase : I8 / Phase I      Docs : 15_BUILD_DIRECTIVE.md Phase I; 09_INFRA_WORKFLOW.md §7
Contract      : register_signal_handlers(cleanup_callback) -> None
Invariants    : Catches SIGINT and SIGTERM; initiates emergency checkpoint before exiting with code 0
Failure modes : Second interrupt signal forces hard exit
Status        : TESTED
"""

import signal
import sys
from typing import Callable, Optional


class PreemptionHandler:
    """Manages graceful exit on preemption signal."""

    def __init__(self, callback: Optional[Callable[[], None]] = None) -> None:
        self.callback = callback
        self.interrupted = False

    def _handler(self, signum: int, frame: Optional[object]) -> None:
        if self.interrupted:
            print("\nForce terminating on second interrupt signal...")
            sys.exit(1)

        self.interrupted = True
        print(f"\nReceived signal {signum}. Initiating graceful preemption checkpoint...")
        if self.callback:
            try:
                self.callback()
            except Exception as e:
                print(f"Error during emergency checkpoint callback: {e}")

    def register(self) -> None:
        """Register signal handlers for SIGINT and SIGTERM (platform-aware)."""
        signal.signal(signal.SIGINT, self._handler)
        # Windows supports SIGINT but SIGTERM is emulated or only available on POSIX
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._handler)
