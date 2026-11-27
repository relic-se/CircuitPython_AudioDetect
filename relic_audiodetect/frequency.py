# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT

import array
import math

from relic_audiodetect import _DTYPE_FLOAT, _prepare_data, np


def _nearest_pow2(value: int, max: int = 32) -> int:
    for i in range(1, max + 1):
        current = 2**i
        if current > value:
            return 2 ** (i - 1)
        elif current == value:
            return value
    return None


def _is_pow2(value: int, max: int = 32) -> bool:
    return _nearest_pow2(value, max) == value


def _fft(data: np.ndarray) -> np.ndarray:
    # Magnitude of the lower half of the spectrum. Calculated from the FFT directly because
    # `ulab.utils.spectrogram` returns incorrect results on builds of ulab without complex number
    # support (ulab 6.12).
    half = len(data) // 2
    result = np.fft.fft(data)
    if isinstance(result, tuple):
        # ulab without complex number support returns the real and imaginary parts
        real, imag = result[0][:half], result[1][:half]
        return np.sqrt(real * real + imag * imag)
    return abs(result[:half])


def _find_peaks(data: np.ndarray, height: float | None = None) -> tuple[float]:
    if height is None:
        height = 7 * np.std(data)

    # A peak is higher than the value before it, and not lower than the value after it. This is
    # done with whole arrays, because a loop over the spectrum takes most of the time otherwise.
    values = data[1:-1]
    values = np.where(values > data[:-2], values, 0)
    values = np.where(values >= data[2:], values, 0)

    # Remove peaks which are less than the height
    values = np.where(values > height, values, 0)

    return tuple([(int(i) + 1, values[i]) for i in np.nonzero(values)[0]])


class Frequency:
    def __init__(self, data_size: int, sample_rate: int):
        if not _is_pow2(data_size):
            raise ValueError("data_size must be a power of 2")

        self._data_size = data_size
        self._sample_rate = sample_rate

        # Hann window. Without it, the samples start and end abruptly, which spreads a tone over
        # the whole spectrum and moves its peak.
        self._window = 0.5 - 0.5 * np.cos(
            np.arange(data_size, dtype=_DTYPE_FLOAT) * (2 * math.pi / data_size)
        )

        self._frequencies = None

    def _get_frequency(self, index: int, data: np.ndarray) -> float:
        # A tone usually lies between two indexes of the spectrum. Through a Hann window its peak
        # is close to a parabola on a logarithmic scale, so the top of the parabola through the
        # peak and its two neighbors is where the tone is.
        offset = 0.0
        if 0 < index < len(data) - 1:
            left, center, right = data[index - 1], data[index], data[index + 1]
            if left > 0 and center > 0 and right > 0:
                left, center, right = math.log(left), math.log(center), math.log(right)
                curve = left - 2 * center + right
                if curve < 0:
                    offset = 0.5 * (left - right) / curve
        return (index + offset) * self._sample_rate / self._data_size

    def process(self, data: list | tuple | array.array | np.ndarray) -> None:
        if len(data) != self._data_size:
            raise ValueError("Data size invalid")

        # Convert data to float
        data = _prepare_data(data)

        # Remove any offset and apply the window
        data = (data - np.mean(data)) * self._window

        # Perform FFT
        data = _fft(data)

        # Find the index of the peaks and their heights
        peaks = _find_peaks(data)

        # Calculate the interpolated frequency of each peak and pack with height
        self._frequencies = tuple(
            [(self._get_frequency(index, data), height) for index, height in peaks]
        )

    @property
    def frequencies(self) -> tuple[float]:
        if not self._frequencies:
            return ()
        return tuple([x[0] for x in self._frequencies])

    @property
    def most_prominent_frequency(self) -> float | None:
        if not self._frequencies:
            return None
        elif len(self._frequencies) == 1:
            return self._frequencies[0][0]
        else:
            return sorted(self._frequencies, key=lambda x: x[1], reverse=True)[0][0]

    @property
    def lowest_frequency(self) -> float | None:
        if not self._frequencies:
            return None
        return self._frequencies[0][0]


def calculate_frequencies(
    data: list | tuple | array.array | np.ndarray, sample_rate: int
) -> tuple[float]:
    data = data[: _nearest_pow2(len(data))]
    detect = Frequency(len(data), sample_rate)
    detect.process(data)
    return detect.frequencies


def calculate_frequency(
    data: list | tuple | array.array | np.ndarray, sample_rate: int
) -> float | None:
    frequencies = calculate_frequencies(data, sample_rate)
    return frequencies[0] if frequencies else None
