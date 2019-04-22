import configparser
import setuptools


ini = configparser.ConfigParser()
ini.read('version.ini')

with open('README.md') as readme:
    long_description = readme.read()

tests_require = ['pytest-cov', 'jupyter', 'beautifulsoup4']

setuptools.setup(
    name=ini['version']['name'],
    version=ini['version']['value'],
    author='Daniel Farré Manzorro',
    author_email='d.farre.m@gmail.com',
    description='Fit computation time series  - curve fitting with scipy.optimize',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://bitbucket.org/coleopter/curve-fits',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'],
    packages=setuptools.find_packages(),
    install_requires=['pandas', 'matplotlib', 'scipy',
                      'hilbert-curve@git+https://bitbucket.org/coleopter/hilbert-curve'],
    setup_requires=['setuptools', 'configparser'],
    tests_require=tests_require,
    extras_require={'dev': ['ipdb', 'ipython', 'jupyter'], 'test': tests_require},
)
