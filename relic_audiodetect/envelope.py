# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT

import array

from relic_audiodetect import _prepare_data, decouple_signal, normalized_rms, np

try:
    from synthio import EnvelopeState
except ModuleNotFoundError:
    class EnvelopeState:
        ATTACK = 2
        SUSTAIN = 1
        RELEASE = 0

class Envelope:

    def __init__(
        self,
        sensitivity: float = 0.25,
        attack: float = 1,  # begins calculation when level (relative to sensitivity) is above this value
        sustain: float = 0.9,  # if level dips below this threshold (relative to sensitivity) and then rises above attack again, it will be interpretted as a new note
        release: float = 0.1,  # ends calculation when level (relative to sensitivity) is below this value
    ):
        self._sensitivity = min(max(sensitivity, 0.0), 1.0)
        self._attack = min(max(attack, 0.0001), 1.0)
        self._sustain = min(max(sustain, 0.0001), 1.0)
        self._release = min(max(release, 0.0001), 1.0)

        self.reset()

    def reset(self) -> None:
        self._level = 0.0
        self._state = None

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        self._sensitivity = min(max(value, 0.0), 1.0)

    @property
    def active(self) -> bool:
        return self._state in {EnvelopeState.ATTACK, EnvelopeState.SUSTAIN}

    @property
    def level(self) -> float:
        return min(self._level / max(1 - self._sensitivity, 0.0001), 1.0)

    @property
    def state(self) -> EnvelopeState|None:
        return self._state

    def update(self, data: list|tuple|array.array|np.ndarray) -> EnvelopeState|None:
        # Convert to float
        data = _prepare_data(data)
        
        # Decouple signal (re-center around mean)
        data = decouple_signal(data)

        # Update level
        self._level = normalized_rms(data)
        
        # Handle state machine
        level = self.level  # Calculate level using sensitivity (in getter)
        if self._state in {None, EnvelopeState.SUSTAIN, EnvelopeState.RELEASE} and level >= self._attack:
            self._state = EnvelopeState.ATTACK
        elif self._state is not EnvelopeState.RELEASE and level <= self._release:
            self._state = EnvelopeState.RELEASE
        elif self._state == EnvelopeState.ATTACK and level <= self._sustain:
            self._state = EnvelopeState.SUSTAIN
        else:
            return None  # no change occurred
        
        # Return new state on change
        return self._state
