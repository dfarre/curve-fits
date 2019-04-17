import unittest

from tests import notebook


class Tests(notebook.NotebookTester, unittest.TestCase):
    notebooks_path = 'notebooks/'
