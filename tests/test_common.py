import decimal
import unittest

from tcp_fits import iround


class TestIround(unittest.TestCase):
    def test_float_odd(self):
        self.assertEqual(iround(0.00115, 2), decimal.Decimal('0.0012'))

    def test_float_even(self):
        self.assertEqual(iround(0.00145, 2), decimal.Decimal('0.0015'))

    def test_text_0exp_odd(self):
        self.assertEqual(iround('1.5', 1), decimal.Decimal('2'))

    def test_text_0exp_even(self):
        self.assertEqual(iround('2.5', 1), decimal.Decimal('3'))