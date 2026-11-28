"""Library test-suite setup.

The ``experiments`` package is installed in the same environment, but library tests
must not use it: ``None`` in ``sys.modules`` makes any ``import experiments`` raise
``ModuleNotFoundError``. Script tests live in ``experiments/tests`` and run in their
own pytest session.
"""

import sys

sys.modules["experiments"] = None
