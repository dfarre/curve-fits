import abc
import re

import numpy

from numpy.polynomial import polynomial
from scipy import optimize

from curve_fits import Eq, Repr, Piecewise, get_exponent, iround, norm


class Measure(Repr, Eq):
    def __init__(self, value, error, unit='', error_to=2):
        if abs(value) == numpy.inf:
            self.value, self.error = None, None
        elif abs(error) == numpy.inf:
            self.value, self.error = iround(value), None
        else:
            precision = get_exponent(value) - get_exponent(error) + error_to
            self.value, self.error = iround(value, precision), iround(error, error_to)

        self.unit = unit

    def eqkey(self):
        return self.value, self.error, self.unit

    def value_pm_error(self):
        return f'{self.value} ± {self.error}'

    def __str__(self):
        return self.value_pm_error() + (f' {self.unit}' if self.unit else '')


class SeriesFit(Repr, Eq, metaclass=abc.ABCMeta):
    def __init__(self, series, initial_coef, method='lm', fraction=0.9, overfit=-1, sigma=10):
        self.initial_coef = self.get_initial_coefficients(initial_coef)
        self.dof = len(self.initial_coef) - 1
        self.method, self.overfit, self.sigma = method, overfit, sigma
        self.xdata, self.ydata = map(numpy.array, (series.index, series))
        par_series = series.sample(int(fraction*self.xdata.shape[0]))
        self.par_xdata, self.par_ydata = map(numpy.array, (par_series.index, par_series))

        self.function, coef, errors, self.residual, self.slope, self.cost = self._fit()
        self.measures = tuple(Measure(value, error) for value, error in zip(coef, errors))

    def _fit(self):
        """"
        Computes the residual variation when adding the remaining data
        from `fraction` to the whole - in order to avoid overfit.
        May raise:
          - RuntimeError: Optimal parameters not found
          - OptimizeWarning: Covariance of the parameters could not be estimated
        """
        function, coef, errors, residual = self.fit(self.par_xdata, self.par_ydata)
        _, _, _, full_residual = self.fit(self.xdata, self.ydata)
        slope = (full_residual - residual) / (self.xdata.shape[0] - self.par_xdata.shape[0])

        return function, coef, errors, residual, slope, self.compute_cost(residual, slope)

    def compute_cost(self, residual, slope):
        return (residual*numpy.exp(self.overfit) +
                abs(slope)*self.sigma*numpy.exp(-self.overfit))  # [sigma] = `x` size

    def fit(self, xdata, ydata, f=None):
        coef, cov = optimize.curve_fit(
            self.evaluate, xdata, ydata, self.initial_coef, method=self.method)
        function = self.make_function(*coef)

        return function, coef, numpy.sqrt(numpy.diag(cov)), norm(ydata - function(xdata))

    def evaluate(self, x, *params):
        return self.make_function(*params)(x)

    @abc.abstractmethod
    def make_function(self, *params):
        """Return the fitted function"""

    @abc.abstractmethod
    def kind(self):
        """Return short string specifier of the fitting function type"""

    def eqkey(self):
        return self.kind(), self.measures

    @staticmethod
    def get_initial_coefficients(coef_spec):
        return [1]*(coef_spec + 1) if isinstance(coef_spec, int) else coef_spec


class PiecewiseFit(Repr, Eq):
    def __init__(self, series, piece_coef, jumps_at, fit_type, **kwargs):
        self.jumps_at = tuple(jumps_at)
        self.heads = (series.index[0],) + self.jumps_at
        self.edges = (0, *(numpy.dot(
            numpy.array([numpy.where(series.index == x, 1, 0) for x in jumps_at]),
            numpy.arange(len(series.index)))), None)
        self.piece_coef = SeriesFit.get_initial_coefficients(piece_coef)
        self.dof = (len(jumps_at) + 1)*len(self.piece_coef)
        self.fits = tuple(fit_type(
            series.iloc[self.edges[i]:self.edges[i+1]], self.piece_coef, **kwargs)
            for i in range(len(jumps_at) + 1))
        self.function = Piecewise(jumps_at, [fit.function for fit in self.fits])

    def __str__(self):
        return ' | '.join([str(fit) for fit in self.fits])

    def kind(self):
        chain = re.sub(rf'^\[{self.heads[0]}\]', '', '-'.join([
            f'[{head}]{fit.kind()}' for head, fit in zip(self.heads, self.fits)]))

        return f'PW:{chain}'

    def eqkey(self):
        return self.jumps_at, self.fits

    @property
    def cost(self):
        return norm(numpy.array([fit.cost for fit in self.fits]))


class PolynomialFit(SeriesFit):
    def __str__(self):
        return ' + '.join(f'({m.value_pm_error()})' + {0: '', 1: 'x'}.get(d, f'x^{d}')
                          for d, m in enumerate(self.measures))

    def kind(self):
        return f'Poly({self.function.degree()})'

    @staticmethod
    def make_function(*coefficients):
        return polynomial.Polynomial(coefficients)
