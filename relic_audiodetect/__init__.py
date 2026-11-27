# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2026 Cooper Dalrymple
#
# SPDX-License-Identifier: MIT
"""
`relic_audiodetect`
================================================================================

CircuitPython helper library to detect frequency, level changes, and notes in audio.


* Author(s): Cooper Dalrymple

Implementation Notes
--------------------

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the supported boards:
  https://circuitpython.org/downloads
"""

# imports

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/relic-se/CircuitPython_AudioDetect.git"

import array

try:
    import ulab.numpy as np
    _DTYPE_FLOAT = np.float
except ModuleNotFoundError:
    import numpy as np
    _DTYPE_FLOAT = np.float32

def _prepare_data(data: list|tuple|array.array|np.ndarray) -> np.ndarray:
    if type(data) is not np.ndarray or data.dtype != _DTYPE_FLOAT:
        # Convert our data to an np.ndarray object with float values ranging from -1.0 to 1.0
        return np.array(data, dtype=_DTYPE_FLOAT) / (2 ** 15)
    else:
        return data

def decouple_signal(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data)

def normalized_rms(data: np.ndarray) -> float:
    return np.std(data)

class MovingAverage:

    def __init__(self, count: int = 3, weighted: bool = False):
        self._count = count
        self._weights = np.arange(count, dtype=np.float) / count / 2 if weighted else None
        self.reset()

    def reset(self) -> None:
        self._items = None
        self._value = None

    def update(self, value: float) -> None:
        self._value = None
        if self._items is None:
            self._items = np.full(self._count, value)
        else:
            self._items[1:] = self._items[:len(self._items)-1]
            self._items[0] = value

    @property
    def value(self) -> float|None:
        if self._value is not None:
            return self._value
        if self._items is None:
            return None
        if self._weights is None:
            return np.mean(self._items)
        else:
            return np.sum(self._items * self._weights)

class MovingMedian:

    def __init__(self, count: int = 5):
        self._count = count
        self.reset()

    def reset(self) -> None:
        self._items = []
        self._value = None

    def update(self, value: float) -> None:
        self._value = None
        self._items.append(value)
        if len(self._items) > self._count:
            self._items.pop(0)

    @property
    def value(self) -> float|None:
        if self._value is None and self._items:
            # The middle of the values. With an even number of values the lower one of the two
            # in the middle is used, so that the result is always a value that was given.
            self._value = sorted(self._items)[(len(self._items) - 1) // 2]
        return self._value
