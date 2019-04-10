import abc

import numpy

from numpy.polynomial import polynomial
from scipy import optimize

from tcp_fits import BaseRepr, get_exponent, iround


class Measure(BaseRepr):
    def __init__(self, value, error, unit='', error_to=2):
        precision = get_exponent(value) - get_exponent(error) + error_to
        self.value = iround(value, precision)
        self.error = iround(error, error_to)
        self.unit = unit

    @property
    def value_pm_error(self):
        return f'{self.value} ± {self.error}'

    def __str__(self):
        return self.value_pm_error + (f' {self.unit}' if self.unit else '')


class Fit(BaseRepr, metaclass=abc.ABCMeta):
    def __init__(self, data, initial_coef, method, data_fraction=0.8):
        self.xdata, self.ydata = map(numpy.array, (data.series.index, data.series))
        self.length = self.xdata.shape[0]
        self.data = data
        self.initial_coef = initial_coef
        self.method = method
        self.partial_fit_length = int(data_fraction*self.length)
        self.function, coef, cov, self.residual, _ = self._fit()
        _, _, _, _, self.residual_slope = self._fit(partial=True)
        errors = numpy.sqrt(numpy.diag(cov))
        self.measures = [Measure(value, error) for value, error in zip(coef, errors)]

    def _compute_residual(self, function, partial=False):
        length = self.partial_fit_length if partial else self.length
        residuals = self.ydata[:length] - function(self.xdata[:length])

        return numpy.sqrt(numpy.inner(residuals, residuals)/length)

    def _fit(self, partial=False):
        """"
        If `partial` compute the residual variation when adding the remaining data,
        from `self.data_fraction` to the 100% (in order to avoid overfitting)
        """
        length = self.partial_fit_length if partial else self.length
        coef, cov = optimize.curve_fit(
            self.evaluate, self.xdata[:length], self.ydata[:length], self.initial_coef,
            method=self.method)
        function = self.make_function(*coef)
        residual = self._compute_residual(function, partial=partial)
        residual_slope = (self._compute_residual(function) - residual
                          ) / (self.length - length) if partial else 0

        return function, coef, cov, residual, residual_slope

    def evaluate(self, x, *params):
        return self.make_function(*params)(x)

    @abc.abstractmethod
    def make_function(self, *params):
        """Return the fitted function"""


class PolynomialFit(Fit):
    def __str__(self):
        return ' + '.join(f'({m.value_pm_error})' + {0: '', 1: 'x'}.get(d, f'x^{d}')
                          for d, m in enumerate(self.measures))

    @staticmethod
    def make_function(*coefficients):
        return polynomial.Polynomial(coefficients)


class SeriesFit(BaseRepr):
    def __init__(self, series):
        self.series = series
        self.best_fit = None

    def __str__(self):
        return f'{self.series}'

    def polynomial_fit(self, *initial_coeff, data_fraction=0.8):
        """
        Polynomial fit of minimum residual slope computed with `data_fraction`.
        `initial_coeff` is a list of arrays of initial coefficients
        """
        return min((PolynomialFit(self, coef, method, data_fraction=data_fraction)
                    for method in {'lm', 'trf', 'dogbox'} for coef in initial_coeff),
                   key=lambda fit: fit.residual_slope)

    def logarithmic_fit(self, *initial_coeff, data_fraction=0.8):
        raise NotImplementedError

    def fit(self, data_fraction=0.8, **kwargs):
        self.best_fit = min((
            getattr(self, f'{k}_fit')(*kwargs[k], data_fraction=data_fraction)
            for k in kwargs), key=lambda fit: fit.residual)

    def plot(self, axes, alpha=1):
        self.series.plot(ax=axes, alpha=alpha)

        if self.best_fit:
            x = numpy.array(self.series.index)
            label = f'{self.series.name}: {self.best_fit}'
            axes.plot(x, self.best_fit.function(x), label=label, zorder=10000)
