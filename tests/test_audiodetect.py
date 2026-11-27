# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT

"""Tests that run on a computer with numpy. Run with `pytest` from the root of the repository."""

import math
import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from relic_audiodetect import MovingMedian
from relic_audiodetect.envelope import Envelope, EnvelopeState
from relic_audiodetect.frequency import Frequency, _fft, _find_peaks, calculate_frequencies, calculate_frequency
from relic_audiodetect.note import Note

SAMPLE_RATE = 44100
CHUNK_SIZE = 2048
BIN_WIDTH = SAMPLE_RATE / CHUNK_SIZE

# Recordings of single guitar notes and the frequency of the note that was played
RECORDINGS = (
    ("guitar-e2-82hz.wav", 82.41, "E2"),
    ("guitar-a2-110hz.wav", 110.00, "A2"),
    ("guitar-d3-147hz.wav", 146.83, "D3"),
    ("guitar-g3-196hz.wav", 196.00, "G3"),
    ("guitar-b3-247hz.wav", 246.94, "B3"),
)


def tone(frequency: float, length: int = CHUNK_SIZE, amplitude: float = 0.5) -> np.ndarray:
    data = amplitude * np.sin(2 * np.pi * frequency * np.arange(length) / SAMPLE_RATE)
    return data.astype(np.float32)


def as_int16(data: np.ndarray):
    return (data * 32767).astype(np.int16)


def cents(frequency: float, reference: float) -> float:
    return 1200 * math.log2(frequency / reference)


def read_recording(filename: str) -> np.ndarray:
    with wave.open(os.path.join(os.path.dirname(__file__), filename), "rb") as file:
        assert file.getframerate() == SAMPLE_RATE
        assert file.getnchannels() == 1
        assert file.getsampwidth() == 2
        data = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2")
    return data.astype(np.float32) / 2**15


def lowest_frequencies(data: np.ndarray, chunk_size: int = CHUNK_SIZE) -> list:
    detect = Frequency(chunk_size, SAMPLE_RATE)
    frequencies = []
    for i in range(len(data) // chunk_size):
        detect.process(data[i * chunk_size : (i + 1) * chunk_size])
        if detect.lowest_frequency:
            frequencies.append(detect.lowest_frequency)
    return frequencies


# Frequency


@pytest.mark.parametrize("frequency", (110, 220, 440, 880, 1000, 2500))
def test_frequency_of_tone(frequency):
    detect = Frequency(CHUNK_SIZE, SAMPLE_RATE)
    detect.process(tone(frequency))
    assert abs(detect.most_prominent_frequency - frequency) < BIN_WIDTH / 2
    assert abs(detect.lowest_frequency - frequency) < BIN_WIDTH / 2


def test_frequency_accepts_integer_samples():
    detect = Frequency(CHUNK_SIZE, SAMPLE_RATE)
    detect.process(as_int16(tone(440)))
    assert abs(detect.most_prominent_frequency - 440) < BIN_WIDTH / 2


def test_frequency_of_two_tones():
    detect = Frequency(CHUNK_SIZE, SAMPLE_RATE)
    detect.process(tone(440, amplitude=0.2) + tone(1320, amplitude=0.5))
    assert len(detect.frequencies) == 2
    assert abs(detect.lowest_frequency - 440) < BIN_WIDTH / 2
    assert abs(detect.most_prominent_frequency - 1320) < BIN_WIDTH / 2


def test_frequency_of_silence():
    detect = Frequency(CHUNK_SIZE, SAMPLE_RATE)
    detect.process(np.zeros(CHUNK_SIZE, dtype=np.float32))
    assert detect.frequencies == ()
    assert detect.lowest_frequency is None
    assert detect.most_prominent_frequency is None


def test_frequencies_before_processing():
    assert Frequency(CHUNK_SIZE, SAMPLE_RATE).frequencies == ()


def test_frequency_data_size():
    with pytest.raises(ValueError):
        Frequency(1000, SAMPLE_RATE)
    with pytest.raises(ValueError):
        Frequency(CHUNK_SIZE, SAMPLE_RATE).process(tone(440, 1024))


@pytest.mark.parametrize("frequency", (82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 440.0, 880.0, 1046.5))
@pytest.mark.parametrize("harmonics", ((1.0,), (1.0, 0.5, 0.3, 0.2)))
def test_frequency_of_tone_is_in_tune(frequency, harmonics):
    """A tone is found to within 10 cents wherever it lies between two indexes of the spectrum."""
    detect = Frequency(CHUNK_SIZE, SAMPLE_RATE)
    time = np.arange(CHUNK_SIZE) / SAMPLE_RATE
    for step in range(16):
        phase = step * math.pi / 8
        data = sum(
            level * np.sin((index + 1) * (2 * np.pi * frequency * time + phase))
            for index, level in enumerate(harmonics)
        )
        detect.process((0.4 * data / sum(harmonics)).astype(np.float32))
        assert abs(cents(detect.lowest_frequency, frequency)) < 10


def test_frequency_ignores_offset():
    detect = Frequency(CHUNK_SIZE, SAMPLE_RATE)
    detect.process(tone(440) + 0.25)
    assert abs(cents(detect.lowest_frequency, 440)) < 10


def reference_peaks(data: np.ndarray) -> list:
    """The peaks of a spectrum, found one value at a time."""
    height = 7 * np.std(data)
    return [
        i
        for i in range(1, len(data) - 1)
        if data[i - 1] < data[i] and data[i] >= data[i + 1] and data[i] > height
    ]


@pytest.mark.parametrize(("filename", "frequency", "name"), RECORDINGS)
def test_peaks_of_recording(filename, frequency, name):
    data = read_recording(filename)
    count = 0
    for i in range(len(data) // CHUNK_SIZE):
        spectrum = _fft(data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE])
        peaks = _find_peaks(spectrum)
        assert [index for index, height in peaks] == reference_peaks(spectrum)
        assert all(height == spectrum[index] for index, height in peaks)
        count += len(peaks)
    assert count > 0


def test_peaks_of_flat_data():
    assert _find_peaks(np.zeros(64, dtype=np.float32)) == ()
    assert _find_peaks(np.ones(64, dtype=np.float32)) == ()


def test_peak_height():
    data = np.zeros(64, dtype=np.float32)
    data[10] = 2.0
    data[20] = 5.0
    assert [index for index, height in _find_peaks(data, height=1.0)] == [10, 20]
    assert [index for index, height in _find_peaks(data, height=3.0)] == [20]


def test_calculate_frequency():
    data = tone(440, 3000)  # is cut down to a power of 2
    assert len(calculate_frequencies(data, SAMPLE_RATE)) == 1
    assert abs(calculate_frequency(data, SAMPLE_RATE) - 440) < BIN_WIDTH / 2


@pytest.mark.parametrize(("filename", "frequency", "name"), RECORDINGS)
def test_frequency_of_recording(filename, frequency, name):
    """The lowest peak is the fundamental of the note, to within one index of the spectrum."""
    detected = np.median(lowest_frequencies(read_recording(filename)))
    assert abs(detected - frequency) < BIN_WIDTH


@pytest.mark.parametrize(("filename", "frequency", "name"), RECORDINGS)
def test_frequency_of_recording_is_in_tune(filename, frequency, name):
    """A tuner needs to be within a few cents. The recordings are assumed to be within 15."""
    detected = np.median(lowest_frequencies(read_recording(filename)))
    assert abs(cents(detected, frequency)) < 25


# Envelope


def test_envelope_states():
    envelope = Envelope(sensitivity=0.9)  # attack at an RMS level of 0.1

    assert envelope.update(np.zeros(CHUNK_SIZE, dtype=np.float32)) is EnvelopeState.RELEASE
    assert not envelope.active

    assert envelope.update(tone(440, amplitude=0.5)) is EnvelopeState.ATTACK
    assert envelope.active
    assert envelope.level == 1.0

    assert envelope.update(tone(440, amplitude=0.5)) is None  # no change

    assert envelope.update(tone(440, amplitude=0.1)) is EnvelopeState.SUSTAIN
    assert envelope.active

    assert envelope.update(tone(440, amplitude=0.5)) is EnvelopeState.ATTACK  # a new note

    assert envelope.update(tone(440, amplitude=0.001)) is EnvelopeState.RELEASE
    assert not envelope.active


def test_envelope_ignores_offset():
    envelope = Envelope(sensitivity=0.9)
    envelope.update(np.full(CHUNK_SIZE, 0.5, dtype=np.float32))
    assert envelope.level < 0.001


def test_envelope_level_follows_sensitivity():
    data = tone(440, amplitude=0.1)  # RMS of about 0.071
    low = Envelope(sensitivity=0.0)
    low.update(data)
    high = Envelope(sensitivity=0.9)
    high.update(data)
    assert low.level == pytest.approx(0.0707, abs=0.002)
    assert high.level == pytest.approx(0.707, abs=0.02)


# Note


def test_note_of_tone():
    note = Note(CHUNK_SIZE, SAMPLE_RATE)
    note.sensitivity = 0.9
    assert note.notename == ""
    assert note.notenum is None

    for _ in range(5):
        note.update(tone(440))
    assert note.active
    assert note.notenum == 69
    assert note.notename == "A4"
    assert abs(note.frequency - 440) < BIN_WIDTH / 2


def test_note_cents():
    note = Note(CHUNK_SIZE, SAMPLE_RATE)
    note.sensitivity = 0.9
    for _ in range(5):
        note.update(tone(440))
    assert note.cents == pytest.approx(cents(note.frequency, 440.0), abs=0.01)


def test_note_frequency_of_second_note():
    note = Note(CHUNK_SIZE, SAMPLE_RATE)
    note.sensitivity = 0.9
    for _ in range(5):
        note.update(tone(440))
    for _ in range(2):
        note.update(np.zeros(CHUNK_SIZE, dtype=np.float32))
    assert not note.active

    # The attack is reported with the frequency of the samples after the ones the note starts in
    assert note.update(tone(220)) is None
    assert note.update(tone(220)) == EnvelopeState.ATTACK
    assert note.notename == "A3"
    assert abs(note.frequency - 220) < BIN_WIDTH / 2


def test_note_attack_waits_for_frequency():
    # Noise is loud enough to start a note, but its spectrum has no peak
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, CHUNK_SIZE).astype(np.float32)
    note = Note(CHUNK_SIZE, SAMPLE_RATE)
    note.sensitivity = 0.9
    note.update(np.zeros(CHUNK_SIZE, dtype=np.float32))

    for _ in range(3):
        assert note.update(noise) is None
        assert note.active
        assert note.notenum is None

    assert note.update(tone(440)) == EnvelopeState.ATTACK
    assert note.notename == "A4"
    assert note.update(tone(440)) is None


def test_note_ignores_single_wrong_frequency():
    note = Note(CHUNK_SIZE, SAMPLE_RATE)
    note.sensitivity = 0.9
    for _ in range(3):
        note.update(tone(220))
    note.update(tone(440))  # as when one update finds the second harmonic instead of the note
    assert note.notename == "A3"
    assert abs(note.frequency - 220) < 1
    note.update(tone(220))
    assert note.notename == "A3"


def test_note_follows_new_frequency():
    note = Note(CHUNK_SIZE, SAMPLE_RATE)
    note.sensitivity = 0.9
    for _ in range(5):
        note.update(tone(220))
    for _ in range(3):
        note.update(tone(440))  # the same level, so the envelope does not start a new note
    assert note.notename == "A4"


@pytest.mark.parametrize(
    ("frequency", "notenum"), ((220.0, 57), (329.63, 64), (440.0, 69), (659.26, 76), (880.0, 81))
)
def test_note_at_attack(frequency, notenum):
    """The note that is reported with the attack is right wherever in the samples it starts."""
    size = 512
    time = np.arange(6 * size) / SAMPLE_RATE
    for phase in range(4):
        played = sum(
            level * np.sin((index + 1) * (2 * np.pi * frequency * time + phase * math.pi / 2))
            for index, level in enumerate((1.0, 0.5, 0.3, 0.2))
        )
        played = (0.42 * played / np.max(np.abs(played))).astype(np.float32)
        for start in range(0, size, size // 16):
            data = np.concatenate([np.zeros(2 * size + start, dtype=np.float32), played])
            note = Note(size, SAMPLE_RATE)
            note.sensitivity = 0.9
            reported = None
            for i in range(len(data) // size):
                if note.update(data[i * size : (i + 1) * size]) == EnvelopeState.ATTACK:
                    reported = note.notenum
                    break
            assert reported == notenum, f"note starts at sample {start} with phase {phase}"


def test_note_keeps_frequency_until_next_attack():
    note = Note(CHUNK_SIZE, SAMPLE_RATE)
    note.sensitivity = 0.9
    for _ in range(4):
        note.update(tone(440))
    for _ in range(2):
        note.update(np.zeros(CHUNK_SIZE, dtype=np.float32))
    assert note.notename == "A4"  # of the note that ended
    note.update(tone(220))
    assert note.notename == "A4"  # the new note has started, but is not known yet
    assert note.update(tone(220)) == EnvelopeState.ATTACK
    assert note.notename == "A3"


def test_note_reset():
    note = Note(CHUNK_SIZE, SAMPLE_RATE)
    note.sensitivity = 0.9
    note.update(tone(440))
    note.reset()
    assert not note.active
    assert note.frequency is None
    assert note.notename == ""


@pytest.mark.xfail(reason="A spectrum of 2048 samples is too coarse for the lowest notes", strict=False)
@pytest.mark.parametrize(("filename", "frequency", "name"), RECORDINGS)
def test_note_of_recording(filename, frequency, name):
    data = read_recording(filename)
    note = Note(CHUNK_SIZE, SAMPLE_RATE)
    note.sensitivity = 0.98
    names = []
    for i in range(len(data) // CHUNK_SIZE):
        note.update(data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE])
        if note.active:
            names.append(note.notename)
    assert names
    assert max(set(names), key=names.count) == name


# MovingMedian


def test_moving_median():
    median = MovingMedian(5)
    assert median.value is None
    median.update(3.0)
    assert median.value == 3.0
    median.update(9.0)
    assert median.value == 3.0  # the lower of the two values in the middle
    median.update(4.0)
    assert median.value == 4.0
    for value in (100.0, 1.0):
        median.update(value)
    assert median.value == 4.0  # of 3, 9, 4, 100, 1
    median.update(5.0)
    assert median.value == 5.0  # of 9, 4, 100, 1, 5: the oldest value is dropped
    median.reset()
    assert median.value is None
