import abc
import decimal
import functools

import numpy

from matplotlib import pyplot


class Repr(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __str__(self):
        """Object's text content"""

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self}>'


class Eq(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eqkey(self):
        """Return hashable key property to compare to others"""

    def __eq__(self, other):
        return self.eqkey() == other.eqkey()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.eqkey())


class Call(Repr):
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __str__(self):
        return f'(*{self.args}, **{self.kwargs})'


class Spec:
    def __init__(self, curve_type, dof, **kwargs):
        self.curve_type, self.dof, self.kwds = curve_type, dof, kwargs


class PyplotShow:
    def __init__(self, **defaults):
        self.defaults = defaults

    def __call__(self, plot_method):
        @functools.wraps(plot_method)
        def wrapper(obj, *args, **kwargs):
            kwds = {**self.defaults, **kwargs}
            figure, axes = pyplot.subplots(figsize=kwds.pop('figsize'))
            plot_method(obj, *args, **{**kwds, 'axes': axes, 'figure': figure})
            pyplot.grid()
            pyplot.show()
            return figure, axes

        return wrapper


def norm(x: numpy.array):
    return numpy.sqrt(numpy.inner(x, x)/x.shape[0])


def get_exponent(number):
    if number in {-numpy.inf, numpy.inf}:
        return numpy.inf

    return int(numpy.format_float_scientific(float(number)).split('e')[1])


def iround(number, to=1):
    """Round to `to` significant digits"""
    exp = get_exponent(number)

    if exp == numpy.inf:
        return decimal.Decimal(number)

    return decimal.Decimal(str(number)).scaleb(-exp).quantize(
        decimal.Decimal('1.' + '0'*(to - 1)), rounding=decimal.ROUND_HALF_UP
    ).scaleb(exp)
