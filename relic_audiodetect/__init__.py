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

All functions and classes which accept audio take either signed 16-bit samples (a `list`,
`tuple`, `array.array` or integer ``ndarray``) or a float ``ndarray`` with values ranging from
-1.0 to 1.0.

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the supported boards with :mod:`ulab` support:
  https://circuitpython.org/downloads

* :mod:`synthio` is used when available, but is not required.

* On CPython, `numpy <https://numpy.org/>`_ is used in place of :mod:`ulab`.
"""

# imports

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/relic-se/CircuitPython_AudioDetect.git"

import array

try:
    import ulab.numpy as np

    _DTYPE_FLOAT = np.float
except ImportError:
    import numpy as np

    _DTYPE_FLOAT = np.float32


def _prepare_data(data: list | tuple | array.array | np.ndarray) -> np.ndarray:
    if type(data) is not np.ndarray or data.dtype != _DTYPE_FLOAT:
        # Convert our data to an np.ndarray object with float values ranging from -1.0 to 1.0
        return np.array(data, dtype=_DTYPE_FLOAT) / (2**15)
    else:
        return data


def decouple_signal(data: np.ndarray) -> np.ndarray:
    """Remove any DC offset from a signal by re-centering it around its mean.

    :param data: The signal as a float ``ndarray``.
    :return: A new ``ndarray`` with a mean of 0.
    """
    return data - np.mean(data)


def normalized_rms(data: np.ndarray) -> float:
    """Calculate the RMS level of a signal, ignoring any DC offset.

    :param data: The signal as a float ``ndarray`` with values from -1.0 to 1.0.
    :return: The RMS level of the signal, from 0.0 to 1.0.
    """
    return np.std(data)


class MovingMedian:
    """Keeps the median of the last ``count`` values given to it. Unlike an average, the median is
    not moved by a single outlier.

    :param int count: The number of values to keep. Defaults to 5.
    """

    def __init__(self, count: int = 5):
        self._count = count
        self.reset()

    def reset(self) -> None:
        """Clear all values."""
        self._items = []
        self._value = None

    def update(self, value: float) -> None:
        """Add a value, removing the oldest value if there are more than ``count``.

        :param float value: The value to add.
        """
        self._value = None
        self._items.append(value)
        if len(self._items) > self._count:
            self._items.pop(0)

    @property
    def value(self) -> float | None:
        """The median of the current values, or ``None`` if there are none. With an even number of
        values, the lower of the two middle values is used."""
        if self._value is None and self._items:
            # The middle of the values. With an even number of values the lower one of the two
            # in the middle is used, so that the result is always a value that was given.
            self._value = sorted(self._items)[(len(self._items) - 1) // 2]
        return self._value
