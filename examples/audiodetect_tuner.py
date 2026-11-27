# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: Unlicense

"""A simple instrument tuner. Prints the name of each note played near the microphone and how many
cents sharp or flat it is.

Written for boards with a built-in PDM microphone, such as the Circuit Playground Bluefruit.
"""

import array

import audiobusio
import board

from relic_audiodetect.envelope import EnvelopeState
from relic_audiodetect.note import Note

SAMPLE_RATE = 16000
DATA_SIZE = 2048  # must be a power of 2, larger sizes detect lower notes more accurately

mic = audiobusio.PDMIn(
    board.MICROPHONE_CLOCK,
    board.MICROPHONE_DATA,
    sample_rate=SAMPLE_RATE,
    bit_depth=16,
)
samples = array.array("H", [0] * DATA_SIZE)

note = Note(DATA_SIZE, SAMPLE_RATE)
note.sensitivity = 0.9  # increase to detect quieter notes

while True:
    mic.record(samples, len(samples))
    state = note.update(samples)

    if state == EnvelopeState.ATTACK:
        print("Note started")
    elif state == EnvelopeState.RELEASE:
        print("Note ended")

    if note.active and note.notenum is not None:
        cents = note.cents
        if abs(cents) < 5:
            tuning = "in tune"
        elif cents > 0:
            tuning = "sharp"
        else:
            tuning = "flat"
        print(f"{note.notename:>7s} {note.frequency:7.2f} Hz {cents:+6.1f} cents ({tuning})")
