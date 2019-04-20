import abc

import numpy

from numpy.polynomial import polynomial

from curve_fits import Repr, Vector, Scale

EXPONENTS = {0: '', 1: '', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹'}


class Curve(Repr, Vector):
    def __init__(self, *curves, shift=0):
        self.curves = curves
        self._add = shift

    def __call__(self, x: numpy.array):
        return self._add + sum([curve(x) for curve in self.curves])

    def add_other(self, curve):
        return Curve(*(self.curves + curve.curves), shift=self._add+curve._add)

    def add_number(self, number):
        return Curve(*self.curves, shift=self._add+number)

    def num_prod(self, number):
        return Curve(*(number*cu for cu in self.curves), shift=number*self._add)

    def __str__(self):
        return ' + '.join(([str(self._add)] if self._add else [])
                          + list(map(str, self.curves)))

    def kind(self):
        return '+'.join(sorted(curve.kind() for curve in self.curves))


class AbstractCurve(Repr, Scale, metaclass=abc.ABCMeta):
    def __init__(self, *parameters, pole=0, norm=1):
        self.parameters, self.pole, self.norm = numpy.array(parameters), pole, norm

    def __call__(self, x: numpy.array):
        return self.evaluate((x - self.pole)/self.norm)

    def __str__(self):
        return self.format(*self.parameters)

    @abc.abstractmethod
    def evaluate(self, s: numpy.array):
        """Normalized function"""

    def kind(self):
        return self.__class__.__name__

    @staticmethod
    def format(*params):
        """Text representation given parameters `params`"""


class NonLinearCurve(AbstractCurve, metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mul = 1

    def num_prod(self, number):
        self._mul *= number
        return self


class XtoA(NonLinearCurve):
    def evaluate(self, s: numpy.array):
        return self._mul*self.parameters[0]*s**self.parameters[1]

    def format(self, *params):
        return f'({self._mul*params[0]})s^({params[1]})'


class Piecewise(NonLinearCurve):
    def __init__(self, jumps_at, curves, **super_kwds):
        super().__init__(**super_kwds)

        self.jumps_at, self.piece_count = jumps_at, len(jumps_at) + 1
        self.curves = curves

    def __str__(self):
        return ' | '.join(list(map(str, self.curves)))

    def kind(self):
        headed = zip(self.jumps_at, self.curves[1:])
        chain = '-'.join([self.curves[0].kind()] + [
            f'[{head}]{curve.kind()}' for head, curve in headed])

        return f'PW:{chain}'

    def evaluate(self, s: numpy.array):
        functions = numpy.dot(self.conditions(s), self.curves)

        return self._mul*numpy.array([f(v) for f, v in zip(functions, s)])

    def conditions(self, s: numpy.array):
        return numpy.where(numpy.array([s < self.jumps_at[0]] + [
            self.jumps_at[i] <= s < self.jumps_at[i+1] for i in range(self.piece_count - 2)
        ] + [s >= self.jumps_at[self.piece_count-2]]).transpose(), 1, 0)


class LinearCurve(AbstractCurve, metaclass=abc.ABCMeta):
    def num_prod(self, number):
        self.parameters *= number
        return self


class Log(LinearCurve):
    def evaluate(self, s: numpy.array):
        return self.parameters[0]*numpy.log(s)

    @staticmethod
    def format(*params):
        return f'({params[0]})log(s)'


class Xlog(LinearCurve):
    def evaluate(self, s: numpy.array):
        return self.parameters[0]*s*numpy.log(s)

    @staticmethod
    def format(*params):
        return f'({params[0]})s·log(s)'


class Polynomial(LinearCurve):
    def evaluate(self, s: numpy.array):
        return polynomial.Polynomial(self.parameters)(s)

    def kind(self):
        return f'Poly({len(self.parameters) - 1})'

    @staticmethod
    def format(*params):
        return ' + '.join([f'({param})' + ('s' if d > 0 else '') + EXPONENTS.get(
            d, f'^{d}') for d, param in enumerate(params)])


class InverseXPolynomial(LinearCurve):
    def evaluate(self, s: numpy.array):
        return polynomial.Polynomial([0] + list(reversed(self.parameters)))(1/s)

    def kind(self):
        return f'Poly(-{len(self.parameters)})'

    @staticmethod
    def format(*params):
        return ' + '.join(reversed([f'({param})/s' + EXPONENTS.get(n + 1, f'^{n + 1}')
                                    for n, param in enumerate(reversed(params))]))
