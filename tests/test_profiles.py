import unittest

from tests import notebook


class ProfilesTests(unittest.TestCase):
    @notebook.Notebook('notebooks/list.ipynb')
    def test_linear_list_methods(self):
        pass
