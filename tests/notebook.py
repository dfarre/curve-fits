import functools
import os
import re
import subprocess

import bs4


class Notebook:
    def __init__(self, path):
        self.html_path = os.path.abspath(re.sub(r'.ipynb$', '.html', path))
        self.path = path

    def __call__(self, method):
        @functools.wraps(method)
        def wrapper(test_case, *args, **kwargs):
            subprocess.Popen([
                'jupyter', 'nbconvert', '--execute', '--allow-errors', self.path]
            ).communicate()
            errors = ', '.join(self.get_error_input_numbers())

            assert not errors, f'Notebook {self.path} {errors} failed - ' \
                f'check file://{self.html_path}'

        return wrapper

    def get_error_input_numbers(self):
        with open(self.html_path) as html:
            soup = bs4.BeautifulSoup(html.read(), features='html.parser')

        return [self.get_error_input_number(error) for error in soup.find_all(
            'div', {'class': 'output_error'})]

    @staticmethod
    def get_error_input_number(error_soup):
        parents = error_soup.parents
        cell = [next(parents) for x in range(4)][-1]
        content = cell.find('div', {'class': 'input_prompt'}).contents[0]

        return content.replace('\xa0', '').replace(':', '')
