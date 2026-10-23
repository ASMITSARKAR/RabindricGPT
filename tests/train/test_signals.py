"""
tests/train/test_signals.py : Tests for OS signal handling and emergency checkpoint save on preemption.
"""

from src.rgpt.train.signals import PreemptionHandler


def test_preemption_signal_triggers_checkpoint() -> None:
    """Preemption signal must invoke emergency state save callback cleanly."""
    called = []

    def emergency_callback():
        called.append(True)

    handler = PreemptionHandler(callback=emergency_callback)
    handler.register()

    # Simulate trigger
    handler._handler(signum=2, frame=None)

    assert handler.interrupted is True
    assert len(called) == 1
