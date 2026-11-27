Simple test
------------

Ensure your device works with this simple test. It finds the frequency of a generated tone, so no
extra hardware is needed.

.. literalinclude:: ../examples/audiodetect_simpletest.py
    :caption: examples/audiodetect_simpletest.py
    :linenos:

Tuner
-----

Detect the notes played near a PDM microphone and how far each one is out of tune, using
:class:`relic_audiodetect.note.Note`.

.. literalinclude:: ../examples/audiodetect_tuner.py
    :caption: examples/audiodetect_tuner.py
    :linenos:

Sound detection
---------------

Turn on an LED while a sound is heard by a PDM microphone, using
:class:`relic_audiodetect.envelope.Envelope`.

.. literalinclude:: ../examples/audiodetect_envelope.py
    :caption: examples/audiodetect_envelope.py
    :linenos:

Frequency plotter
-----------------

Continuously print the frequencies heard by a PDM microphone in a format which can be plotted by
the Mu editor, using :class:`relic_audiodetect.frequency.Frequency`.

.. literalinclude:: ../examples/audiodetect_frequency.py
    :caption: examples/audiodetect_frequency.py
    :linenos:
