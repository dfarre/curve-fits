import itertools
import timeit

import pandas
import numpy

from matplotlib import pyplot

from curve_fits import PyplotShow
from curve_fits import fits

GOLDEN_RATIO = 1.61803398875


class FittingFrame:
    def __init__(self, *args, label='', **kwargs):
        self.fraction = kwargs.pop('fraction', 0.9)
        self.overfit = kwargs.pop('overfit', -1)
        self.data = pandas.DataFrame(*args, **kwargs)
        self._fits = {key: set() for key in list(self.data)}
        self.label = label

    def fit(self, **calls):
        for key, call_sequence in calls.items():
            for call in call_sequence:
                call.kwargs.update({
                    'overfit': self.overfit, 'fraction': self.fraction})
                self._fits[key].update(getattr(self, f'{call.name}_fits')(
                    key, *call.args, **call.kwargs))

    def fit_all_with(self, *calls):
        self.fit(**dict.fromkeys(self.data, calls))

    def polynomial_fits(self, key, *initial_coeff, **kwargs):
        return {min((fits.PolynomialFit(
            self.data[key], coef, method=method, **kwargs)
            for method in {'lm', 'trf', 'dogbox'}), key=lambda fit: fit.cost)
            for coef in initial_coeff}

    def polynomial_stair_fits(self, key, piece_coef_specs, jumps_at, **kwargs):
        return {min((fits.PiecewiseFit(
            self.data[key], coef_spec, jumps, fits.PolynomialFit,
            method=method, **kwargs) for method in {'lm', 'trf', 'dogbox'}),
            key=lambda fit: fit.cost
        ) for coef_spec in piece_coef_specs for jumps in jumps_at}

    def best_fit(self, key):
        return min(self._fits[key], key=lambda fit: fit.cost)

    def best_fits(self, limit=2):
        data = numpy.array(list(itertools.chain(*([
            [key, fit.cost, fit.kind(), fit, fit.dof] for fit in sorted(
                fits, key=lambda fit: fit.cost)[:limit]]
            for key, fits in self._fits.items()))))
        index = pandas.MultiIndex.from_arrays([data[:, 0], data[:, 1]], names=['key', 'cost'])

        return pandas.DataFrame(data[:, 2:], columns=['kind', 'fit', 'DOF'], index=index)

    @PyplotShow(figsize=(5*2*GOLDEN_RATIO, 5))
    def plot_costs(self, key, limit=None, rotation=90, **kwargs):
        fits = self.best_fits(limit).loc[key]
        x = numpy.arange(len(fits))
        pyplot.bar(x, fits.index)
        pyplot.xticks(x, fits.kind, rotation=rotation)
        pyplot.title('Fit costs - ascending')

    @PyplotShow(figsize=(8*GOLDEN_RATIO, 8), style='o-', alpha=0.5)
    def plot(self, limit=2, **kwargs):
        axes = kwargs.pop('axes')
        self.data.plot(ax=axes, **kwargs)

        for (key, cost), (kind, fit, dof) in self.best_fits(limit).iterrows():
            x = numpy.array(self.data.index)
            axes.plot(x, fit.function(x), label=f'{key}: {kind}')

        axes.legend()
        axes.yaxis.set_label_text(self.label)


class TimeComplexityProfile(FittingFrame):
    def __init__(self, init_calls, *method_calls, module='', loops=10, **kwargs):
        def time_method(init_call, method_call):
            mseconds = 10**3*timeit.Timer(
                (f'{module}.' if module else '') + f'{init_call}.{method_call}',
                f'import {module}' if module else 'pass').timeit(loops)

            return mseconds / loops

        super().__init__({mc.name.strip('_'): [
            time_method(ic, mc) for ic in init_calls] for mc in method_calls},
            label='Time (ms/loop)', **kwargs)
