import os
import re
import subprocess

import bs4


class NotebookTester:
    def __init_subclass__(cls):
        cls.test_notebooks = {
            m.groups()[0]: os.path.join(cls.notebooks_path, m.string) for m in filter(
                None, map(lambda n: re.match(r'^(.+)-test.ipynb$', n),
                          os.listdir(cls.notebooks_path)))}

        for name, path in cls.test_notebooks.items():
            html_path = os.path.abspath(re.sub(r'.ipynb$', '.html', path))

            def test(self):
                subprocess.Popen([
                    'jupyter', 'nbconvert', '--execute', '--allow-errors', path
                ]).communicate()

                with open(html_path) as html:
                    soup = bs4.BeautifulSoup(html.read(), features='html.parser')

                errors = ', '.join(list(cls.yield_error_input_numbers(soup)))

                assert not errors, f'Notebook {path} {errors} failed - ' \
                    f'check file://{html_path}'

            setattr(cls, f'test_{name}', test)

    @staticmethod
    def yield_error_input_numbers(soup):
        for error_soup in soup.find_all('div', {'class': 'output_error'}):
            parents = error_soup.parents
            cell = [next(parents) for x in range(4)][-1]
            content = cell.find('div', {'class': 'input_prompt'}).contents[0]

            yield content.replace('\xa0', '').replace(':', '')
