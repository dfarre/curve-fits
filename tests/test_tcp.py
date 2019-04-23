import unittest

from ipynb_tests import tester


class Tests(tester.NotebookTester, unittest.TestCase):
    notebooks_path = 'notebooks/'
