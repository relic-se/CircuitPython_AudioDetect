# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT

from random import sample
import sys
sys.path.insert(0, "../")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import os

from relic_audiodetect.frequency import Frequency
from relic_audiodetect.envelope import Envelope

CHUNK_SIZE = 2048

# Clean
for filename in os.listdir("./"):
    if not filename.startswith(".") and filename.endswith(".png"):
        os.remove(filename)

def audio_test(data: np.ndarray, sample_rate: int, name: str) -> None:
    plt.plot(
        np.arange(len(data), dtype=np.float32) / sample_rate,
        data
    )
    plt.ylabel("Level")
    plt.xlabel("Time (s)")
    plt.savefig(name + "audio.png")
    plt.clf()

def frequency_test(data: np.ndarray, sample_rate: int, name: str) -> None:
    detect = Frequency(CHUNK_SIZE, sample_rate)
    frequency_data = []
    for i in range(len(data) // CHUNK_SIZE):
        chunk = data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        detect.process(chunk)
        lowest, most_prominent = detect.lowest_frequency, detect.most_prominent_frequency
        frequency_data.append([
            lowest if lowest else 0.0,
            most_prominent if most_prominent else 0.0
        ])

    plt.plot(
        np.arange(len(frequency_data), dtype=np.float32) / sample_rate * CHUNK_SIZE,
        frequency_data
    )
    plt.ylabel("Hz")
    plt.xlabel("Time (s)")
    plt.savefig(name + "frequency.png")
    plt.clf()

def envelope_test(data: np.ndarray, sample_rate: int, name: str) -> None:
    detect = Envelope(0.98)
    
    frequency_data = []
    for i in range(len(data) // CHUNK_SIZE):
        chunk = data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        detect.update(chunk)
        
        frequency_data.append([
            detect.level,
            detect.state
        ])

    plt.plot(
        np.arange(len(frequency_data), dtype=np.float32) / sample_rate * CHUNK_SIZE,
        frequency_data
    )
    plt.ylabel("Level / State")
    plt.xlabel("Time (s)")
    plt.savefig(name + "envelope.png")
    plt.clf()

def run_tests(data: np.ndarray, sample_rate: int, name: str = "") -> None:
    if len(name):
        name += "-"

    audio_test(data, sample_rate, name)
    frequency_test(data, sample_rate, name)
    envelope_test(data, sample_rate, name)

for filename in os.listdir("./"):
    if not filename.startswith(".") and filename.endswith(".wav"):
        # Read wav audio data
        sample_rate, data = wavfile.read(filename)

        # convert to numpy float array
        data = np.array(data, dtype=np.float32) / (2 ** 15)

        # begin tests
        run_tests(data, sample_rate, filename[:-len(".wav")])
