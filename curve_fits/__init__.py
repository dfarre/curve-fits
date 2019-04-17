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
    def __init__(self, name, *args, **kwargs):
        self.name, self.args, self.kwargs = name, args, kwargs

    def __str__(self):
        return f'{self.name}(*{self.args}, **{self.kwargs})'


class Piecewise:
    def __init__(self, jumps_at, functions):
        self.jumps_at, self.piece_count = jumps_at, len(jumps_at) + 1
        self.functions = numpy.array(functions)

        if not self.functions.shape[0] == self.piece_count:
            raise AssertionError('1 function <-> 1 piece required')

    def __call__(self, x: numpy.array):
        functions = numpy.dot(self.conditions(x), self.functions)

        return numpy.array([f(v) for f, v in zip(functions, x)])

    def conditions(self, x):
        return numpy.where(numpy.array([x < self.jumps_at[0]] + [
            self.jumps_at[i] <= x < self.jumps_at[i+1] for i in range(self.piece_count - 2)
        ] + [x >= self.jumps_at[self.piece_count-2]]).transpose(), 1, 0)


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

        return wrapper


def norm(x: numpy.array):
    return numpy.sqrt(numpy.inner(x, x)/x.shape[0])


def get_exponent(number):
    if number in {-numpy.inf, numpy.inf}:
        return numpy.inf

    return int(numpy.format_float_scientific(float(number)).split('e')[1])


def iround(number, to=1):
    exp = get_exponent(number)

    if exp == numpy.inf:
        return decimal.Decimal(number)

    return decimal.Decimal(str(number)).scaleb(-exp).quantize(
        decimal.Decimal('1.' + '0'*(to - 1)), rounding=decimal.ROUND_HALF_UP
    ).scaleb(exp)
