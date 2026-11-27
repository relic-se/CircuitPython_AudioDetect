# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: Unlicense

"""Turn on an LED while a sound is heard by the microphone, such as a clap or a whistle.

Written for boards with a built-in PDM microphone, such as the Circuit Playground Bluefruit.
"""

import array

import audiobusio
import board
import digitalio

from relic_audiodetect.envelope import Envelope, EnvelopeState

SAMPLE_RATE = 16000
DATA_SIZE = 256  # smaller sizes respond more quickly

mic = audiobusio.PDMIn(
    board.MICROPHONE_CLOCK,
    board.MICROPHONE_DATA,
    sample_rate=SAMPLE_RATE,
    bit_depth=16,
)
samples = array.array("H", [0] * DATA_SIZE)

led = digitalio.DigitalInOut(board.LED)
led.switch_to_output()

envelope = Envelope(sensitivity=0.9)  # increase sensitivity to detect quieter sounds

while True:
    mic.record(samples, len(samples))
    state = envelope.update(samples)

    if state == EnvelopeState.ATTACK:
        print("Sound started")
    elif state == EnvelopeState.RELEASE:
        print("Sound ended")

    led.value = envelope.active
