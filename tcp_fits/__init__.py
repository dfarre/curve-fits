import decimal

import numpy
import pandas


def arange_frame(l, r, step, *functions):
    x = numpy.arange(l, r, step)

    return pandas.DataFrame({f.__name__: f(x) for f in functions}, index=x)


def get_exponent(number):
    return int(numpy.format_float_scientific(float(number)).split('e')[1])


def iround(number, to=1):
    exp = get_exponent(number)

    return decimal.Decimal(str(number)).scaleb(-exp).quantize(
        decimal.Decimal('1.' + '0'*(to - 1)), rounding=decimal.ROUND_HALF_UP).scaleb(exp)


class BaseRepr:
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self}>'
