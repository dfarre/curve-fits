import abc

import numpy

from scipy import optimize

from curve_fits import Eq, Repr, get_exponent, iround, norm

from hilbert.curves import PiecewiseCurve


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


class Fit(Repr, Eq, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, series, space, *args, **kwargs):
        """Fit series - set `self.curve` in `space`"""

    @classmethod
    def make_fits(cls, series, space, *init_calls):
        return {cls(series, space, *call.args, **call.kwargs) for call in init_calls}


class CurveFit(Fit):
    def __init__(self, series, space, *curve_specs, initial_params=None,
                 method='lm', fraction=0.9, overfit=-1, sigma=10, **curve_defaults):
        self.space = space
        self.curve_dofs = tuple(spec.dof for spec in curve_specs)
        self.dof = sum(self.curve_dofs)
        self.initial_params = initial_params
        self.curve_specs = curve_specs

        if initial_params is None:
            self.initial_params = [1]*self.dof
        elif len(initial_params) != self.dof:
            raise AssertionError(f'Expected {self.dof} DOF from curve specs - '
                                 f'got {len(initial_params)} parameters')

        self.method, self.overfit, self.sigma = method, overfit, sigma
        self.xdata, self.ydata = map(numpy.array, (series.index, series))
        par_series = series.sample(int(fraction*self.xdata.shape[0]))
        self.par_xdata, self.par_ydata = map(numpy.array, (par_series.index, par_series))
        self.curve_defaults = curve_defaults

        self.curve, coef, errors, self.residual, self.slope, self.cost = self._fit()
        self.measures = tuple(Measure(value, error) for value, error in zip(coef, errors))

    def __str__(self):
        return ' + '.join([curve.format(*measures) for curve, measures in zip(
            self.curve.curves, self.split_params(self.measures))])

    def evaluate(self, x, *parameters):
        return self.make_curve(parameters)(x)

    def make_curve(self, parameters):
        return self.space(*(
            spec.curve_type(*params, **{**self.curve_defaults, **spec.kwds})
            for spec, params in zip(self.curve_specs, self.split_params(parameters))))

    def split_params(self, params):
        return [params[sum(self.curve_dofs[:i]):
                       sum(self.curve_dofs[:i]) + self.curve_dofs[i]]
                for i in range(len(self.curve_dofs))]

    def eqkey(self):
        return self.curve.kind(), self.measures

    def _fit(self):
        """"
        Computes the residual variation when adding the remaining data
        from `fraction` to the whole - in order to handle overfitting.
        May raise:
          - RuntimeError: Optimal parameters not found
          - OptimizeWarning: Covariance of the parameters could not be estimated
        """
        curve, coef, errors, residual = self.fit(self.par_xdata, self.par_ydata)
        _, _, _, full_residual = self.fit(self.xdata, self.ydata)
        slope = (full_residual - residual) / (self.xdata.shape[0] - self.par_xdata.shape[0])

        return curve, coef, errors, residual, slope, self.compute_cost(residual, slope)

    def compute_cost(self, residual, slope):
        return (residual*numpy.exp(self.overfit) +
                abs(slope)*self.sigma*numpy.exp(-self.overfit))  # [sigma] = `x` size

    def fit(self, xdata, ydata, f=None):
        coef, cov = optimize.curve_fit(
            self.evaluate, xdata, ydata, self.initial_params, method=self.method)
        curve = self.make_curve(coef)

        return curve, coef, numpy.sqrt(numpy.diag(cov)), norm(ydata - curve(xdata))

    @classmethod
    def make_fits(cls, series, space, *init_calls):
        return {min((cls(series, space, *call.args, method=method, **call.kwargs)
                     for method in {'lm', 'trf', 'dogbox'}), key=lambda fit: fit.cost)
                for call in init_calls}


class PiecewiseFit(Fit):
    def __init__(self, series, space, jumps_at, *curve_fit_calls,
                 fraction=0.9, overfit=-1, sigma=1):
        if not len(curve_fit_calls) == len(jumps_at) + 1:
            raise AssertionError('1 curve fit call per piece required')

        self.space = space
        self.jumps_at = tuple(jumps_at)
        self.heads = (series.index[0],) + self.jumps_at
        self.edges = (0, *(numpy.dot(
            numpy.array([numpy.where(series.index == x, 1, 0) for x in jumps_at]),
            numpy.arange(len(series.index)))), None)
        fit_kwargs = dict(fraction=fraction, overfit=overfit, sigma=sigma)
        self.fits = tuple(CurveFit(
            series.iloc[self.edges[i]:self.edges[i+1]], self.space, *call.args, **{
                **call.kwargs, **fit_kwargs}) for i, call in enumerate(curve_fit_calls))
        self.curve = self.space(PiecewiseCurve(jumps_at, [fit.curve for fit in self.fits]))
        self.dof = sum([fit.dof for fit in self.fits])

    def __str__(self):
        return ' | '.join(list(map(str, self.fits)))

    def eqkey(self):
        return self.jumps_at, self.fits

    @property
    def cost(self):
        return norm(numpy.array([fit.cost for fit in self.fits]))
