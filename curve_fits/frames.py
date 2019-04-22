import itertools
import timeit

import pandas
import numpy

from matplotlib import pyplot

from hilbert import spaces

from curve_fits import PyplotShow

from curve_fits import fits

GOLDEN_RATIO = 1.61803398875


class FittingFrame:
    def __init__(self, *args, label='', fraction=0.9, overfit=-1, sigma=1, **kwargs):
        self.fit_kwargs = dict(fraction=fraction, overfit=overfit, sigma=sigma)
        self.data = pandas.DataFrame(*args, **kwargs)
        self.space = spaces.LebesgueCurveSpace(spaces.Reals.from_index(self.data.index))
        self._fits = {key: set() for key in list(self.data)}
        self.label = label

    def fit(self, **curve_fit_calls):
        self.type_fit(fits.CurveFit, curve_fit_calls)

    def piecewise_fit(self, **piecewise_fit_calls):
        self.type_fit(fits.PiecewiseFit, piecewise_fit_calls)

    def type_fit(self, fit_type, calls):
        for key, call_sequence in calls.items():
            for call in call_sequence:
                call.kwargs.update(self.fit_kwargs)

            self._fits[key].update(fit_type.make_fits(
                self.data[key], self.space, *call_sequence))

    def fit_all_with(self, *curve_fit_calls, piecewise=()):
        self.fit(**dict.fromkeys(self.data, curve_fit_calls))
        self.piecewise_fit(**dict.fromkeys(self.data, piecewise))

    def best_fit(self, key):
        return min(self._fits[key], key=lambda fit: fit.cost)

    def best_fits(self, limit=2):
        data = numpy.array(list(itertools.chain(*([
            [key, fit.cost, fit.curve.kind(), fit, fit.dof] for fit in sorted(
                fits, key=lambda fit: fit.cost)[:limit]]
            for key, fits in self._fits.items()))))
        index = pandas.MultiIndex.from_arrays([data[:, 0], data[:, 1]], names=['key', 'cost'])

        return pandas.DataFrame(data[:, 2:], columns=['kind', 'fit', 'DOF'], index=index)

    @PyplotShow(figsize=(5*2*GOLDEN_RATIO, 5))
    def plot_costs(self, key, limit=None, rotation=90, **kwargs):
        fits = self.best_fits(limit).loc[key]
        x = numpy.arange(len(fits))
        pyplot.bar(x, fits.index, **kwargs)
        pyplot.xticks(x, fits.kind, rotation=rotation)
        pyplot.title('Fit costs - ascending')

    @PyplotShow(figsize=(8*GOLDEN_RATIO, 8), style='o-', alpha=0.5)
    def plot(self, limit=2, **kwargs):
        axes = kwargs.pop('axes')
        self.data.plot(ax=axes, **kwargs)

        for (key, cost), (kind, fit, dof) in self.best_fits(limit).iterrows():
            x = numpy.array(self.data.index)
            axes.plot(x, fit.curve(x), label=f'{key}: {kind}')

        axes.legend()
        axes.yaxis.set_label_text(self.label)


class TimeComplexityProfile(FittingFrame):
    def __init__(self, type_name, init_calls, method_calls, module='', loops=10, **kwargs):
        def time_method(name, init_call, call):
            mseconds = 10**3*timeit.Timer(
                (f'{module}.' if module else '') + f'{type_name}{init_call}.{name}{call}',
                f'import {module}' if module else 'pass').timeit(loops)

            return mseconds / loops

        super().__init__({name.strip('_'): [
            time_method(name, ic, mc) for ic in init_calls]
            for name, mc in method_calls.items()}, label='Time (ms/loop)', **kwargs)
