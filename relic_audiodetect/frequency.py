# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT

import array
import math

from relic_audiodetect import _prepare_data, np, _DTYPE_FLOAT

def _nearest_pow2(value: int, max: int = 32) -> int:
    for i in range(1, max + 1):
        current = 2 ** i
        if current > value:
            return 2 ** (i - 1)
        elif current == value:
            return value
    return None

def _is_pow2(value: int, max: int = 32) -> bool:
    return _nearest_pow2(value, max) == value

try:
    import ulab  # noqa: F401 (the fallback below is used if ulab is not available)

    def _fft(data: np.ndarray) -> np.ndarray:
        # Calculate the magnitude from the FFT directly. `ulab.utils.spectrogram` returns
        # incorrect results on builds of ulab without complex number support (ulab 6.12).
        real, imag = np.fft.fft(data)
        real, imag = real[:len(data) // 2], imag[:len(data) // 2]
        return np.sqrt(real * real + imag * imag)

    def _fftfreq(size: int, spacing: float = 1.0) -> np.ndarray:
        # we're ignoring imaginary half
        return np.arange(size // 2, dtype=np.int16) / (size * spacing)

    def _local_maxima_1d(data: np.ndarray) -> tuple:
        midpoints = []

        i = 1
        while i < len(data) - 1:
            # Test if previous sample is smaller
            if data[i - 1] < data[i]:

                # Find next sample that is unequal to x[i]
                j = i + 1  # Index to look ahead of current sample
                while j < len(data) - 1 and data[j] == data[i]:
                    j += 1

                # Maxima is found if next unequal sample is smaller than x[i]
                if data[j] < data[i]:
                    midpoints.append((i + j - 1) // 2)

                    # Skip samples that can't be maximum
                    i = j
            i += 1

        return tuple(midpoints)

    def _find_peaks(data: np.ndarray, height: float|None = None) -> tuple[float]:
        peaks = _local_maxima_1d(data)
        peak_heights = np.array([data[i] for i in peaks], dtype=_DTYPE_FLOAT)

        if height is None:
            height = 7 * np.std(data)

        # Remove elements which are less than the height
        peak_heights = np.where(peak_heights > height, peak_heights, 0)
        valid_peaks = np.nonzero(peak_heights)[0]
        return tuple([
            (peaks[i], peak_heights[i]) for i in valid_peaks
        ])

except ModuleNotFoundError:
    from scipy.fft import fft
    from scipy.signal import find_peaks

    def _fft(data: np.ndarray) -> np.ndarray:
        return np.abs(fft(data, axis=0)[:len(data) // 2])

    def _fftfreq(size: int, spacing: float = 1.0) -> np.ndarray:
        return np.fft.fftfreq(size, spacing)[:size // 2]

    def _find_peaks(data: np.ndarray, height: float|None = None) -> tuple[float]:
        if height is None:
            height = 7 * np.std(data)
        peaks = find_peaks(data, height=height)
        return tuple([
            (peaks[0][i], peaks[1]["peak_heights"][i]) for i in range(len(peaks[0]))
        ])

class Frequency:

    def __init__(self, data_size: int, sample_rate: int, window_size: int = 17):
        if not _is_pow2(data_size):
            raise ValueError("data_size must be a power of 2")
        
        self._data_size = data_size
        self._sample_rate = sample_rate
        self._window_size = min(window_size + 1 if (window_size % 2) == 0 else window_size, 3)
        self._half_window_size = self._window_size // 2

        # Calculate fftfreq plot
        self._fftfreq = _fftfreq(self._data_size, 1 / self._sample_rate)

        # Linear distribution of indexes used to calculate weighted mean
        self._window_dist = np.arange(self._window_size, dtype=np.int16)

        self._frequencies = None

    def _get_frequency(self, index: int, data: np.ndarray) -> float|None:
        # Skip weighted mean if we're at the edge
        if index <= self._half_window_size or index >= self._data_size - self._half_window_size - 1:
            return self._fftfreq[index]

        # Isolate the window area
        window = data[index - self._half_window_size:index + self._half_window_size + 1]

        # Get the center index using weighted mean
        window_sum = np.sum(window)
        if window_sum <= 0:
            return None
        weighted_index = np.sum(window * self._window_dist) / window_sum

        # Adjust index by weighted mean
        index += weighted_index - (self._half_window_size)

        # Check if we're centered at a frequency index
        index_floor, index_ceil = math.floor(index), math.ceil(index)
        if index_floor == index_ceil:
            return self._fftfreq[index]

        # Perform linear interpolation to adjust detected frequency
        value_floor, value_ceil = self._fftfreq[index_floor], self._fftfreq[index_ceil]
        return (index - index_floor) * (value_ceil - value_floor) + value_floor

    def process(self, data: list|tuple|array.array|np.ndarray) -> None:
        if len(data) != self._data_size:
            raise ValueError("Data size invalid")

        # Convert data to float
        data = _prepare_data(data)

        # Perform FFT
        data = _fft(data)

        # Find the index of the peaks and their heights
        peaks = _find_peaks(data)

        # Calculate frequencies (with windowed weighted mean) for each peak and pack with height
        self._frequencies = [(self._get_frequency(index, data), height) for index, height in peaks]

        # Remove any values with `None`
        self._frequencies = tuple(filter(lambda x: x[0] is not None, self._frequencies))

    @property
    def frequencies(self) -> tuple[float]:
        if not self._frequencies:
            return ()
        return tuple([x[0] for x in self._frequencies])

    @property
    def most_prominent_frequency(self) -> float|None:
        if not self._frequencies:
            return None
        elif len(self._frequencies) == 1:
            return self._frequencies[0][0]
        else:
            return sorted(self._frequencies, key=lambda x: x[1], reverse=True)[0][0]

    @property
    def lowest_frequency(self) -> float|None:
        if not self._frequencies:
            return None
        return self._frequencies[0][0]

def calculate_frequencies(data: list|tuple|array.array|np.ndarray, sample_rate: int) -> tuple[float]:
    data = data[:_nearest_pow2(len(data))]
    detect = Frequency(len(data), sample_rate)
    detect.process(data)
    return detect.frequencies


def calculate_frequency(data: list|tuple|array.array|np.ndarray, sample_rate: int) -> float|None:
    frequencies = calculate_frequencies(data, sample_rate)
    return frequencies[0] if frequencies else None
