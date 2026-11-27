# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: Unlicense

"""Find the frequency of a generated tone. No extra hardware is needed."""

import math

from relic_audiodetect.frequency import calculate_frequency

SAMPLE_RATE = 16000
DATA_SIZE = 1024  # must be a power of 2
TONE = 440.0  # A4

# Generate a sine wave as signed 16-bit samples
samples = [int(16384 * math.sin(2 * math.pi * TONE * i / SAMPLE_RATE)) for i in range(DATA_SIZE)]

frequency = calculate_frequency(samples, SAMPLE_RATE)
print(f"Generated {TONE:.1f} Hz, detected {frequency:.1f} Hz")
