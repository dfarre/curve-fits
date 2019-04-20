import unittest

import numpy

from curve_fits import iround

from curve_fits.curves import Curve, Log, Xlog, Polynomial, InverseXPolynomial, XtoA


class CurveAlgebraTests(unittest.TestCase):
    def test_linear(self):
        u = numpy.array([3, -1, 2])
        v = numpy.array([Curve(Xlog(1/2, pole=-1), InverseXPolynomial(3.14)),
                         +Curve(Log(1, pole=-1)), 0])
        w = numpy.array([1, 0, Curve(Polynomial(1, 1))])
        curve = numpy.dot(u - v, w)

        assert repr(curve) == '<Curve: 3 + (-0.5)s*log(s) + (-3.14)/s + (2) + (2)s>'
        assert iround(curve(numpy.array([2.34]))[0], 7) == iround(numpy.array([
            3 - 0.5*(2.34 + 1)*numpy.log(2.34 + 1) - 3.14/2.34 + 2 + 2*2.34])[0], 7)

    def test_nonlinear(self):
        curve = +2*Curve(XtoA(1/2, 6/5))/5 - 1

        assert repr(curve) == '<Curve: -1.0 + (0.2)s^(1.2)>'
        assert iround(curve(numpy.array([0.11]))[0], 7) == iround(numpy.array([
            -1 + 0.2*0.11**1.2])[0], 7)
