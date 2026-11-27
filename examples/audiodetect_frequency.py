# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: Unlicense

"""Continuously print the lowest and most prominent frequencies heard by the microphone. The output
can be plotted with the Mu editor's plotter.

Written for boards with a built-in PDM microphone, such as the Circuit Playground Bluefruit.
"""

import array

import audiobusio
import board

from relic_audiodetect.frequency import Frequency

SAMPLE_RATE = 16000
DATA_SIZE = 1024  # must be a power of 2

mic = audiobusio.PDMIn(
    board.MICROPHONE_CLOCK,
    board.MICROPHONE_DATA,
    sample_rate=SAMPLE_RATE,
    bit_depth=16,
)
samples = array.array("H", [0] * DATA_SIZE)

detect = Frequency(DATA_SIZE, SAMPLE_RATE)

while True:
    mic.record(samples, len(samples))
    detect.process(samples)
    print((detect.lowest_frequency or 0.0, detect.most_prominent_frequency or 0.0))
