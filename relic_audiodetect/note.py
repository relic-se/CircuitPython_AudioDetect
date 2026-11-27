# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT

import array
import math

from relic_audiodetect import MovingAverage, np, _prepare_data
from relic_audiodetect.envelope import Envelope, EnvelopeState
from relic_audiodetect.frequency import Frequency

try:
    from synthio import midi_to_hz
except ModuleNotFoundError:
    def midi_to_hz(notenum: int) -> float:
        return 440.0 * (2.0 ** ((notenum - 69.0) / 12.0))

_LOG2_A4 = math.log(440, 2)
_NOTE_NAMES = ["A", "A#/Bb", "B", "C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab"]

class Note:

    def __init__(self, data_size: int, sample_rate: int, average_count: int = 5):
        self._envelope = Envelope()
        self._frequency = Frequency(data_size, sample_rate)
        self._frequency_value = MovingAverage(average_count)

        self.reset()

    def reset(self) -> None:
        self._envelope.reset()
        self._frequency_value.reset()
        self._notenum = self._notename = self._cents = None

    @property
    def sensitivity(self) -> float:
        return self._envelope.sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        self._envelope.sensitivity = value

    @property
    def active(self) -> bool:
        return self._envelope.active

    @property
    def level(self) -> float:
        return self._envelope.level

    @property
    def state(self) -> EnvelopeState|None:
        return self._envelope.state

    @property
    def frequency(self) -> float|None:
        return self._frequency_value.value

    @property
    def notenum(self) -> int|None:
        if self._notenum is not None:
            return self._notenum
        frequency = self._frequency_value.value
        if frequency is None or frequency <= 0.0:
            return None
        self._notenum = round(12 * (math.log(frequency, 2) - _LOG2_A4) + 69)  # Calculate MIDI note value
        return self._notenum

    @property
    def notename(self) -> str:
        if self._notename is not None:
            return self._notename
        if (notenum := self.notenum) is None:
            return ""
        self._notename = "{:s}{:d}".format(_NOTE_NAMES[(notenum - 21) % 12], (notenum - 12) // 12) 
        return self._notename

    @property
    def cents(self) -> float|None:
        if self._cents is not None:
            return self._cents
        if (notenum := self.notenum) is None:
            return None
        self._cents = 1200.0 * math.log(self._frequency_value.value / midi_to_hz(notenum), 2)
        return self._cents

    def update(self, data: list|tuple|array.array|np.ndarray) -> EnvelopeState|None:
        self._notenum = self._notename = self._cents = None  # Go ahead and dump our cached values

        # Convert to float
        data = _prepare_data(data)

        # Update envelope
        state = self._envelope.update(data)

        # The frequency of a new note is not averaged with the frequency of the previous note
        if state == EnvelopeState.ATTACK:
            self._frequency_value.reset()

        # Process signal if we're active
        if self._envelope.active:

            # Detect frequencies
            self._frequency.process(data)
            frequency = self._frequency.lowest_frequency
            if not frequency:
                return None  # exit just in case something went wrong

            self._frequency_value.update(frequency)  # Use frequency with largest peak

        return state
