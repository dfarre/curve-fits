import importlib
import time

import pandas

from matplotlib import pyplot

from tcp_fits import BaseRepr
from tcp_fits import series_fits


def import_function(path):
    if '.' not in path:
        path = f'example_functions.{path}'

    module_path, function_name = path.rsplit('.', 1)

    return getattr(importlib.import_module(module_path), function_name)


class OneArgProfile(series_fits.SeriesFit):
    def __init__(self, function, arguments, index_name=None):
        self.function = function if callable(function) else import_function(function)
        super().__init__(self._make_series(arguments, index_name))

    def __str__(self):
        return f'{self.function.__name__} - {self.series.index.name}'

    def _make_series(self, arguments, index_name):
        series = pandas.Series((self.compute_time(arg) for arg in arguments),
                               name=self.function.__name__)
        series.index.name = index_name

        return series

    def compute_time(self, argument):
        initial_time = time.time()
        self.function(argument)

        return time.time() - initial_time

    def polyfit(self, max_degree, data_fraction=0.8):
        self.fit(data_fraction, polynomial=([1]*n for n in range(1, max_degree + 2)))


class OneArgComparison(BaseRepr):
    def __init__(self, arguments, obj, *methods, index_name=None):
        self.index_name = index_name
        self.profiles = [OneArgProfile(getattr(obj, m), arguments, index_name)
                         for m in methods]

    def __str__(self):
        function_names = ', '.join(p.function.__name__ for p in self.profiles)

        return f'{function_names} - {self.index_name}'

    def polyfit(self, max_degree, data_fraction=0.8):
        for profile in self.profiles:
            profile.polyfit(max_degree, data_fraction)

    def show(self, min_alpha=0.7, figsize=None):
        figure, axes = pyplot.subplots(figsize=figsize)

        for n, profile in enumerate(self.profiles):
            profile.plot(axes, alpha=(1 - (1 - min_alpha)*n/(len(self.profiles) - 1))
                         if len(self.profiles) > 1 else 1)

        axes.set(ylabel='Time (s)')
        axes.legend()
        pyplot.show()
