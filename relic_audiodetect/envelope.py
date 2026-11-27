# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT
"""
`relic_audiodetect.envelope`
================================================================================

Detect the start and end of sounds by following the level of an audio signal.

* Author(s): Cooper Dalrymple
"""

import array

from relic_audiodetect import _prepare_data, decouple_signal, normalized_rms, np

try:
    from synthio import EnvelopeState
except ImportError:

    class EnvelopeState:
        """Stand-in for :class:`synthio.EnvelopeState` when :mod:`synthio` is not available."""

        ATTACK = 2
        SUSTAIN = 1
        RELEASE = 0


class Envelope:
    """Follows the level of an audio signal and reports when a sound starts (attack), settles
    (sustain), and ends (release).

    The thresholds ``attack``, ``sustain`` and ``release`` are compared against :attr:`level`,
    which is the RMS level of the signal scaled by ``sensitivity``.

    :param float sensitivity: How sensitive detection is to quiet signals, from 0.0 to 1.0. Higher
        values detect quieter sounds. Defaults to 0.25.
    :param float attack: A sound starts when the level rises to or above this value. Defaults to
        1.0.
    :param float sustain: After an attack, the state moves to sustain once the level falls to or
        below this value. If the level then rises to ``attack`` again, it is treated as a new sound.
        Defaults to 0.9.
    :param float release: A sound ends when the level falls to or below this value. Defaults to
        0.1.
    """

    def __init__(
        self,
        sensitivity: float = 0.25,
        attack: float = 1,
        sustain: float = 0.9,
        release: float = 0.1,
    ):
        self._sensitivity = min(max(sensitivity, 0.0), 1.0)
        self._attack = min(max(attack, 0.0001), 1.0)
        self._sustain = min(max(sustain, 0.0001), 1.0)
        self._release = min(max(release, 0.0001), 1.0)

        self.reset()

    def reset(self) -> None:
        """Clear the current level and state."""
        self._level = 0.0
        self._state = None

    @property
    def sensitivity(self) -> float:
        """How sensitive detection is to quiet signals, from 0.0 to 1.0. Higher values detect
        quieter sounds."""
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        self._sensitivity = min(max(value, 0.0), 1.0)

    @property
    def active(self) -> bool:
        """Whether a sound is currently being detected (the state is attack or sustain)."""
        return self._state in {EnvelopeState.ATTACK, EnvelopeState.SUSTAIN}

    @property
    def level(self) -> float:
        """The level of the last update scaled by :attr:`sensitivity`, from 0.0 to 1.0."""
        return min(self._level / max(1 - self._sensitivity, 0.0001), 1.0)

    @property
    def state(self) -> EnvelopeState | None:
        """The current state of the envelope, or ``None`` before any sound has been detected."""
        return self._state

    def update(self, data: list | tuple | array.array | np.ndarray) -> EnvelopeState | None:
        """Process a block of audio samples and update the envelope.

        :param data: Signed 16-bit audio samples, or a float :class:`ulab.numpy.ndarray` with
            values from -1.0 to 1.0.
        :return: The new state if it changed during this update, otherwise ``None``.
        """
        # Convert to float
        data = _prepare_data(data)

        # Decouple signal (re-center around mean)
        data = decouple_signal(data)

        # Update level
        self._level = normalized_rms(data)

        # Handle state machine
        level = self.level  # Calculate level using sensitivity (in getter)
        if (
            self._state in {None, EnvelopeState.SUSTAIN, EnvelopeState.RELEASE}
            and level >= self._attack
        ):
            self._state = EnvelopeState.ATTACK
        elif self._state is not EnvelopeState.RELEASE and level <= self._release:
            self._state = EnvelopeState.RELEASE
        elif self._state == EnvelopeState.ATTACK and level <= self._sustain:
            self._state = EnvelopeState.SUSTAIN
        else:
            return None  # no change occurred

        # Return new state on change
        return self._state
