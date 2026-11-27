Introduction
============


.. image:: https://readthedocs.org/projects/circuitpython-audiodetect/badge/?version=latest
    :target: https://circuitpython-audiodetect.readthedocs.io/
    :alt: Documentation Status



.. image:: https://img.shields.io/discord/327254708534116352.svg
    :target: https://adafru.it/discord
    :alt: Discord


.. image:: https://github.com/relic-se/CircuitPython_AudioDetect/workflows/Build%20CI/badge.svg
    :target: https://github.com/relic-se/CircuitPython_AudioDetect/actions
    :alt: Build Status


.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Code Style: Ruff

CircuitPython helper library to detect frequency, level changes, and notes in audio.

The library is made up of three parts which can be used on their own or together:

* ``relic_audiodetect.frequency``: find the frequencies within a block of audio samples using a
  windowed FFT with interpolation between the points of the spectrum.
* ``relic_audiodetect.envelope``: follow the level of an audio signal to detect when a sound
  starts (attack) and ends (release).
* ``relic_audiodetect.note``: combine both to detect musical notes, their MIDI note number, name,
  and how many cents they are out of tune, which is everything needed for an instrument tuner.

Audio can be given as signed or unsigned 16-bit samples, such as the ``array.array`` filled by
``audiobusio.PDMIn.record``, or as a float ``ulab.numpy.ndarray`` with values from -1.0 to 1.0.


Dependencies
=============
This driver depends on:

* `Adafruit CircuitPython <https://github.com/adafruit/circuitpython>`_ built with
  `ulab <https://docs.circuitpython.org/en/latest/shared-bindings/ulab/index.html>`_
* ``synthio`` is used when it is available, but is not required.

On CPython, `numpy <https://numpy.org/>`_ is used in place of ``ulab``.

Please ensure all dependencies are available on the CircuitPython filesystem.
This is easily achieved by downloading
`the Adafruit library and driver bundle <https://circuitpython.org/libraries>`_
or individual libraries can be installed using
`circup <https://github.com/adafruit/circup>`_.

Installing from PyPI
=====================

On supported GNU/Linux systems like the Raspberry Pi, you can install the driver locally `from
PyPI <https://pypi.org/project/circuitpython-audiodetect/>`_.
To install for current user:

.. code-block:: shell

    pip3 install circuitpython-audiodetect

To install system-wide (this may be required in some cases):

.. code-block:: shell

    sudo pip3 install circuitpython-audiodetect

To install in a virtual environment in your current project:

.. code-block:: shell

    mkdir project-name && cd project-name
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install circuitpython-audiodetect

Installing to a Connected CircuitPython Device with Circup
==========================================================

Make sure that you have ``circup`` installed in your Python environment.
Install it with the following command if necessary:

.. code-block:: shell

    pip3 install circup

With ``circup`` installed and your CircuitPython device connected use the
following command to install:

.. code-block:: shell

    circup install relic_audiodetect

Or the following command to update an existing version:

.. code-block:: shell

    circup update

Usage Example
=============

Find the frequency of a block of samples:

.. code-block:: python

    import math
    from relic_audiodetect.frequency import calculate_frequency

    SAMPLE_RATE = 16000
    samples = [int(16384 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)) for i in range(1024)]
    print(calculate_frequency(samples, SAMPLE_RATE))  # about 440.0

Detect notes played near a PDM microphone, such as the one on the Circuit Playground Bluefruit:

.. code-block:: python

    import array
    import audiobusio
    import board
    from relic_audiodetect.note import Note

    SAMPLE_RATE = 16000
    DATA_SIZE = 2048

    mic = audiobusio.PDMIn(
        board.MICROPHONE_CLOCK, board.MICROPHONE_DATA, sample_rate=SAMPLE_RATE, bit_depth=16
    )
    samples = array.array("H", [0] * DATA_SIZE)

    note = Note(DATA_SIZE, SAMPLE_RATE)
    note.sensitivity = 0.9

    while True:
        mic.record(samples, len(samples))
        note.update(samples)
        if note.active and note.notenum is not None:
            print(note.notename, note.frequency, note.cents)

More examples can be found in the
`examples folder <https://github.com/relic-se/CircuitPython_AudioDetect/tree/main/examples>`_
and in the
`documentation <https://circuitpython-audiodetect.readthedocs.io/en/latest/examples.html>`_.

Documentation
=============
API documentation for this library can be found on `Read the Docs <https://circuitpython-audiodetect.readthedocs.io/>`_.

For information on building library documentation, please check out
`this guide <https://learn.adafruit.com/creating-and-sharing-a-circuitpython-library/sharing-our-docs-on-readthedocs#sphinx-5-1>`_.

Contributing
============

Contributions are welcome! Please read our `Code of Conduct
<https://github.com/relic-se/CircuitPython_AudioDetect/blob/HEAD/CODE_OF_CONDUCT.md>`_
before contributing to help this project stay welcoming.
