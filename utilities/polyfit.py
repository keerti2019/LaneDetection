import numpy as np


class PolyFit:
    def __init__(self, degree, use_cache=False, cache_size=10):
        self.degree = degree
        self.fit_values_cache = []
        self.use_cache = use_cache
        self.cache_size = cache_size
        if use_cache:
            for i in range(degree + 1):
                self.fit_values_cache.append([])

    def perform_poly_fit(self, x_fit_coordinates, y_fit_coordinates, plot_y):
        fit = np.polyfit(y_fit_coordinates, x_fit_coordinates, self.degree)
        mean_fit_values = []

        if self.use_cache:
            for i in range(self.degree + 1):
                self.fit_values_cache[i].append(fit[i])
            for i in range(self.degree + 1):
                mean_fit_values.append(np.mean(self.fit_values_cache[i][-self.cache_size:]))
        else:
            for i in range(self.degree + 1):
                mean_fit_values.append(fit[i])

        fit_equation = 0
        coefficient = self.degree
        for i in range(self.degree + 1):
            fit_equation = (mean_fit_values[i] * plot_y ** coefficient) + fit_equation
            coefficient = coefficient - 1
        return fit_equation
