import abc
import itertools

import pandas
import numpy

from matplotlib import pyplot
from numpy.polynomial import polynomial
from scipy import optimize

from tcp_fits import Eq, Repr, get_exponent, iround

GOLDEN_RATIO = 1.61803398875


class Measure(Repr, Eq):
    def __init__(self, value, error, unit='', error_to=2):
        precision = get_exponent(value) - get_exponent(error) + error_to
        self.value = iround(value, precision)
        self.error = iround(error, error_to)
        self.unit = unit

    def eqkey(self):
        return self.value, self.error, self.unit

    def value_pm_error(self):
        return f'{self.value} ± {self.error}'

    def __str__(self):
        return self.value_pm_error() + (f' {self.unit}' if self.unit else '')


class SeriesFit(Repr, Eq, metaclass=abc.ABCMeta):
    def __init__(self, series, initial_coef, method, fraction=0.9, overfit=-1):
        self.initial_coef = ([1]*(initial_coef + 1) if isinstance(initial_coef, int)
                             else initial_coef)
        self.degree = len(self.initial_coef) - 1
        self.method, self.overfit = method, overfit
        self.xdata, self.ydata = map(numpy.array, (series.index, series))
        par_series = series.sample(int(fraction*self.xdata.shape[0]))
        self.par_xdata, self.par_ydata = map(numpy.array, (par_series.index, par_series))

        self.function, coef, cov, self.cost = self._fit()
        errors = numpy.sqrt(numpy.diag(cov))
        self.measures = tuple(Measure(value, error) for value, error in zip(coef, errors))

    def eqkey(self):
        return self.measures

    def _fit(self):
        """"
        Computes the residual variation when adding the remaining data
        from `fraction` to the whole - in order to avoid overfit
        """
        function, coef, cov, residual = self.fit(partial=True)
        _, _, _, full_residual = self.fit(partial=False)
        slope = (full_residual - residual) / (self.xdata.shape[0] - self.par_xdata.shape[0])

        return function, coef, cov, self.compute_cost(residual, slope)

    @staticmethod
    def compute_residual(function, xdata, ydata):
        residuals = ydata - function(xdata)

        return numpy.sqrt(numpy.inner(residuals, residuals)/len(residuals))

    def compute_cost(self, residual, slope):
        return residual*numpy.exp(self.overfit) + (
            self.degree + 1)*abs(slope)*numpy.exp(-self.overfit)/self.xdata.shape[0]

    def fit(self, partial=False):
        """"
        May raise RuntimeError: Optimal parameters not found.
        """
        xdata = self.par_xdata if partial else self.xdata
        ydata = self.par_ydata if partial else self.ydata
        coef, cov = optimize.curve_fit(
            self.evaluate, xdata, ydata, self.initial_coef, method=self.method)
        function = self.make_function(*coef)
        residual = self.compute_residual(function, xdata, ydata)

        return function, coef, cov, residual

    def evaluate(self, x, *params):
        return self.make_function(*params)(x)

    @abc.abstractmethod
    def make_function(self, *params):
        """Return the fitted function"""


class PolynomialFit(SeriesFit):
    def __str__(self):
        return ' + '.join(f'({m.value_pm_error()})' + {0: '', 1: 'x'}.get(d, f'x^{d}')
                          for d, m in enumerate(self.measures))

    @staticmethod
    def make_function(*coefficients):
        return polynomial.Polynomial(coefficients)


class FittingFrame:
    def __init__(self, *args, label='', **kwargs):
        self.fraction = kwargs.pop('fraction', 0.9)
        self.overfit = kwargs.pop('overfit', -1)
        self.data = pandas.DataFrame(*args, **kwargs)
        self._fits = {key: set() for key in list(self.data)}
        self.label = label

    @property
    def fits(self):
        return pandas.DataFrame(list(itertools.chain(*([[
            key, fit, fit.degree, fit.cost] for fit in sorted(
            fits, key=lambda fit: fit.cost)] for key, fits in self._fits.items()
            if fits))), columns=['key', 'fit', 'degree', 'cost'])

    @property
    def best_fits(self):
        index = [k for k, v in self._fits.items() if v]

        return pandas.DataFrame([[
            fit, fit.degree, fit.cost] for fit in (
                min(self._fits[k], key=lambda fit: fit.cost) for k in index)
            ], columns=['fit', 'degree', 'cost'], index=index)

    def fit(self, **calls):
        for key, call_sequence in calls.items():
            for call in call_sequence:
                call.kwargs.update({
                    'overfit': self.overfit, 'fraction': self.fraction})
                self._fits[key].update(getattr(self, f'{call.name}_fits')(
                    key, *call.args, **call.kwargs))

    def fit_all_with(self, *calls):
        self.fit(**dict.fromkeys(self.data, calls))

    def polynomial_fits(self, key, *initial_coeff, **kwargs):
        """
        Polynomial fits of minimum cost.
        `initial_coeff` is a list of arrays of initial coefficients
        """
        return {min((PolynomialFit(self.data[key], coef, method, **kwargs)
                     for method in {'lm', 'trf', 'dogbox'}), key=lambda fit: fit.cost)
                for coef in initial_coeff}

    def show(self, figsize=(8*GOLDEN_RATIO, 8), **kwargs):
        figure, axes = pyplot.subplots(figsize=figsize)
        self.data.plot(ax=axes, **kwargs)

        for key, (fit, degree, cost) in self.best_fits.iterrows():
            self.plot_fit_curve(key, fit, axes)

        axes.legend()
        axes.yaxis.set_label_text(self.label)
        pyplot.show()

    def plot_fit_curve(self, key, fit, axes):
        x = numpy.array(self.data.index)
        axes.plot(x, fit.function(x), label=f'{self.data[key].name}: {fit}', zorder=1000)
