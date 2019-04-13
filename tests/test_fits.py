import unittest

from tests import notebook


class FitsTests(unittest.TestCase):
    @notebook.Notebook('notebooks/overfitting.ipynb')
    def test_overfit_handling(self):
        pass

    @notebook.Notebook('notebooks/polynomials.ipynb')
    def test_polynomial_fits(self):
        pass
