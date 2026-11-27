# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT

"""
`relic_audiodetect.note`
================================================================================

Detect musical notes, their pitch and their tuning within an audio signal.

* Author(s): Cooper Dalrymple
"""

import array
import math

from relic_audiodetect import MovingMedian, _prepare_data, np
from relic_audiodetect.envelope import Envelope, EnvelopeState
from relic_audiodetect.frequency import Frequency

try:
    from synthio import midi_to_hz
except ImportError:

    def midi_to_hz(notenum: int) -> float:
        return 440.0 * (2.0 ** ((notenum - 69.0) / 12.0))


_LOG2_A4 = math.log(440, 2)
_NOTE_NAMES = ["A", "A#/Bb", "B", "C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab"]


class Note:
    """Detects musical notes within a stream of audio. It combines an
    :class:`~relic_audiodetect.envelope.Envelope` to find when notes start and end, a
    :class:`~relic_audiodetect.frequency.Frequency` to find their pitch, and a
    :class:`~relic_audiodetect.MovingMedian` to smooth the pitch over time.

    Call :meth:`update` with each new block of samples. The pitch of the current note is then
    available from :attr:`frequency`, :attr:`notenum`, :attr:`notename` and :attr:`cents`.

    :param int data_size: The number of samples in each block given to :meth:`update`. Must be
        a power of 2. Larger blocks detect lower notes more accurately, but respond more slowly.
    :param int sample_rate: The sample rate of the audio in Hz.
    :param int median_count: The number of updates the frequency is smoothed over. Defaults to 5.
    :param int attack_updates: The number of updates between the start of a note and the report
        of its attack. If each block of samples overlaps with the block before it, increase this
        so that the frequency is measured from samples which the note fills. Defaults to 1.
    """

    def __init__(
        self,
        data_size: int,
        sample_rate: int,
        median_count: int = 5,
        attack_updates: int = 1,
    ):
        self._envelope = Envelope()
        self._frequency = Frequency(data_size, sample_rate)
        # The median of the last frequencies. Unlike an average, it is not moved by a single
        # update that found a harmonic or noise instead of the note.
        self._frequency_value = MovingMedian(median_count)

        # The frequency of a note is taken from samples that the note fills. When every update
        # gets new samples only, that is the update after the one in which the note starts. When
        # the samples of an update overlap with those of the update before, it takes as many
        # updates as it takes to replace most of the samples.
        self._attack_updates = max(int(attack_updates), 1)

        self.reset()

    def reset(self) -> None:
        """Clear the current note and envelope state."""
        self._envelope.reset()
        self._frequency_value.reset()
        self._attack_wait = 0  # updates left until the attack of the current note is reported
        self._notenum = self._notename = self._cents = None

    @property
    def sensitivity(self) -> float:
        """How sensitive detection is to quiet notes, from 0.0 to 1.0. Higher values detect
        quieter notes. See :attr:`relic_audiodetect.envelope.Envelope.sensitivity`."""
        return self._envelope.sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        self._envelope.sensitivity = value

    @property
    def active(self) -> bool:
        """Whether a note is currently being detected."""
        return self._envelope.active

    @property
    def level(self) -> float:
        """The level of the last update scaled by :attr:`sensitivity`, from 0.0 to 1.0."""
        return self._envelope.level

    @property
    def state(self) -> EnvelopeState | None:
        """The current state of the envelope, or ``None`` before any note has been detected."""
        return self._envelope.state

    @property
    def frequency(self) -> float | None:
        """The smoothed frequency of the current note in Hz, or ``None`` if it is not known."""
        return self._frequency_value.value

    @property
    def notenum(self) -> int | None:
        """The MIDI note number nearest to :attr:`frequency`, or ``None`` if it is not known."""
        if self._notenum is not None:
            return self._notenum
        frequency = self._frequency_value.value
        if frequency is None or frequency <= 0.0:
            return None
        # Calculate MIDI note value
        self._notenum = round(12 * (math.log(frequency, 2) - _LOG2_A4) + 69)
        return self._notenum

    @property
    def notename(self) -> str:
        """The name and octave of the note nearest to :attr:`frequency`, such as ``"A4"`` or
        ``"C#/Db3"``, or an empty string if it is not known."""
        if self._notename is not None:
            return self._notename
        if (notenum := self.notenum) is None:
            return ""
        self._notename = f"{_NOTE_NAMES[(notenum - 21) % 12]:s}{(notenum - 12) // 12:d}"
        return self._notename

    @property
    def cents(self) -> float | None:
        """How far :attr:`frequency` is from the pitch of :attr:`notenum` in cents (hundredths of
        a semitone), from -50.0 to 50.0, or ``None`` if it is not known. Positive values are
        sharp and negative values are flat."""
        if self._cents is not None:
            return self._cents
        if (notenum := self.notenum) is None:
            return None
        self._cents = 1200.0 * math.log(self._frequency_value.value / midi_to_hz(notenum), 2)
        return self._cents

    def update(self, data: list | tuple | array.array | np.ndarray) -> EnvelopeState | None:
        """Process a block of audio samples.

        :param data: ``data_size`` signed 16-bit audio samples, or a float ``ndarray`` with
            values from -1.0 to 1.0.
        :return: The new state if it changed during this update, otherwise ``None``. The attack
            of a note is reported once its frequency is known, which may be a few updates after
            the note starts.
        """
        self._notenum = self._notename = self._cents = None  # Go ahead and dump our cached values

        # Convert to float
        data = _prepare_data(data)

        # Update envelope
        state = self._envelope.update(data)

        if state == EnvelopeState.ATTACK:
            # The frequency of a note is not taken from the samples in which it starts. The note
            # only fills a part of them, and what is found in them is often another note. The
            # attack is reported with the frequency of the samples that follow.
            self._attack_wait = self._attack_updates
            return None
        elif state == EnvelopeState.RELEASE:
            self._attack_wait = 0

        # Process signal if we're active
        if self._envelope.active:
            # Wait for the samples to hold enough of a new note, without the cost of looking for
            # its frequency in the meantime
            if self._attack_wait > 1:
                self._attack_wait -= 1
                return None

            # Detect frequencies
            self._frequency.process(data)
            frequency = self._frequency.lowest_frequency
            if frequency:
                if self._attack_wait:
                    # The frequency of a new note is not smoothed with that of the previous note
                    self._frequency_value.reset()
                self._frequency_value.update(frequency)  # Use frequency with largest peak

            # The attack of a note is reported once its frequency is known
            if self._attack_wait:
                if not frequency:
                    return None
                self._attack_wait = 0
                return EnvelopeState.ATTACK

        return state
