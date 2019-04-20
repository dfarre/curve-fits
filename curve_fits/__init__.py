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


class Scale(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def num_prod(self, number):
        """Return the product"""

    def __mul__(self, number):
        if not number:
            return 0
        elif number == 1:
            return self
        elif isinstance(number, (int, float)):
            return self.num_prod(number)
        else:
            raise NotImplementedError(f'Product by {number}')

    def __rmul__(self, number):
        return self.__mul__(number)

    def __truediv__(self, number):
        return self.__mul__(1/number)

    def __rtruediv__(self, number):
        raise NotImplementedError(f'{self} is not /-invertible')


class Vector(Scale, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_other(self, other):
        """Add other vector"""

    @abc.abstractmethod
    def add_number(self, number):
        """Add numeric - interpreted as other vector"""

    def __add__(self, other):
        if not other:
            return self
        elif isinstance(other, self.__class__):
            return self.add_other(other)
        elif isinstance(other, (int, float)):
            return self.add_number(other)
        else:
            raise NotImplementedError(f'Addition to {other}')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __pos__(self):
        return self


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
    exp = get_exponent(number)

    if exp == numpy.inf:
        return decimal.Decimal(number)

    return decimal.Decimal(str(number)).scaleb(-exp).quantize(
        decimal.Decimal('1.' + '0'*(to - 1)), rounding=decimal.ROUND_HALF_UP
    ).scaleb(exp)
