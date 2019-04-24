import configparser
import setuptools


ini = configparser.ConfigParser()
ini.read('version.ini')

with open('README.md') as readme:
    long_description = readme.read()

tests_require = ['pytest-cov', 'ipynb-tests']

setuptools.setup(
    name=ini['version']['name'],
    version=ini['version']['value'],
    author='Daniel Farré Manzorro',
    author_email='d.farre.m@gmail.com',
    description='Curve fitting with scipy.optimize - fit computation time series',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://bitbucket.org/coleopter/curve-fits',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'],
    packages=setuptools.find_packages(),
    install_requires=['pandas', 'matplotlib', 'scipy', 'hilbert-curve'],
    setup_requires=['setuptools', 'configparser'],
    tests_require=tests_require,
    extras_require={'dev': ['ipdb', 'ipython', 'jupyter'], 'test': tests_require},
)
