import timeit

from tcp_fits import series_fits


class TypeProfile(series_fits.FittingFrame):
    def __init__(self, init_calls, *method_calls, module='', loops=10, **kwargs):
        def time_method(init_call, method_call):
            mseconds = 10**3*timeit.Timer(
                (f'{module}.' if module else '') + f'{init_call}.{method_call}',
                f'import {module}' if module else 'pass').timeit(loops)

            return mseconds / loops

        super().__init__({mc.name.strip('_'): [
            time_method(ic, mc) for ic in init_calls] for mc in method_calls},
            label='Time (ms/loop)', **kwargs)
