import decimal
import unittest

import numpy

from curve_fits import fits


class MeasureTests(unittest.TestCase):
    def test_undefined_measure__inf_value(self):
        measure = fits.Measure(-numpy.inf, 'whatever', 'meters')

        assert measure.value is None
        assert measure.error is None
        assert str(measure) == 'None ± None meters'

    def test_undefined_measure__inf_error(self):
        measure = fits.Measure(15.067, numpy.inf, 'sec')

        assert measure.value == decimal.Decimal('2E+1')
        assert measure.error is None
        assert str(measure) == '2E+1 ± None sec'
