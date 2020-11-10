import numpy as np
import pandas as pd

from scipy.interpolate import splder, splev, splrep

def build_cdf(some_vector):
    cdf = pd.DataFrame({'COUNT': pd.DataFrame({'VALUES': some_vector}).groupby('VALUES')['VALUES'].count().sort_index(ascending=True).cumsum()})
    cdf['VALUES'] = cdf.index
    return cdf

def area_item(left_wall, right_wall, step, treshold=None, include_rate=0.5):
    area = 0.5 * (left_wall + right_wall) * step
    if treshold is None:
        return area
    else:
        return {
                0: 0,
                1: include_rate * area,
                2: area
        }[int(left_wall <= treshold) + int(right_wall <= treshold)]

def full_area(cdf_fraction, step, treshold, include_rate):
    area_items = []
    for every_index in range(0, len(cdf_fraction) - 1):
        walls = [cdf_fraction[every_index], cdf_fraction[every_index + 1]]
        area_items.append(area_item(walls[0], walls[-1], step, treshold, include_rate))
    return sum(area_items)

spline_by_distribution = lambda some_cdf, some_interval: splrep(some_cdf['VALUES'], some_cdf['COUNT'], xb=some_interval[0], xe=some_interval[-1], k=3)
cdf_fraction = lambda some_spline, some_fraction: np.maximum(splev(some_fraction, some_spline, ext=1), 0)

class pulse:
    def __init__(self, distribution_1, distribution_2=None, n_steps=1000, include_rate=0.5):
        self.distribution_1 = distribution_1
        self.distribution_2 = distribution_2
        if distribution_2 is None:
            self.interval = [distribution_1.min(), distribution_1.max()]
        else:
            self.interval = [min([distribution_1.min(), distribution_2.min()]), max([distribution_1.max(), distribution_2.max()])]
        self.n_steps = n_steps
        self.include_rate = include_rate
        self.fraction = np.linspace(self.interval[0], self.interval[-1], n_steps)
        self.step = self.fraction[1] - self.fraction[0]
        self.cdf_1 = build_cdf(self.distribution_1)
        self.cdf_spline_1 = spline_by_distribution(self.cdf_1, self.interval)
        self.spline_1 = splder(self.cdf_spline_1, 1)
        self.density_fraction_1 = cdf_fraction(self.spline_1, self.fraction)
        self.area_1 = full_area(self.density_fraction_1, self.step, None, self.include_rate)
        # self.scaled_cdf_1 = self.cdf_1
        # self.scaled_cdf_1['COUNT'] = self.scaled_cdf_1['COUNT'] / self.area_1
        self.scaled_density_fraction_1 = self.density_fraction_1 / self.area_1
        if distribution_2 is None:
            self.cdf_2 = None
            self.spline_2 = None
            self.density_fraction_2 = None
            self.area_2 = None
            # self.scaled_cdf_2 = None
            self.scaled_density_fraction_2 = None
        else:
            self.cdf_2 = build_cdf(self.distribution_2)
            self.cdf_spline_2 = spline_by_distribution(self.cdf_2, self.interval)
            self.spline_2 = splder(self.cdf_spline_2, 1)
            self.density_fraction_2 = cdf_fraction(self.spline_2, self.fraction)
            self.area_2 = full_area(self.density_fraction_2, self.step, None, self.include_rate)
            # self.scaled_cdf_2 = self.cdf_2
            # self.scaled_cdf_2['COUNT'] = self.scaled_cdf_2['COUNT'] / self.area_2
            self.scaled_density_fraction_2 = self.density_fraction_2 / self.area_2

    def spline_p_value(self, spline, cdf_fraction, area, some_value):
        treshold = splev([some_value], spline, ext=1)[0]
        return full_area(cdf_fraction, self.step, treshold, self.include_rate) / area

    def p_value(self, some_value):
        p1 = self.spline_p_value(self.spline_1, self.density_fraction_1, self.area_1, some_value)
        if self.distribution_2 is not None:
            p2 = self.spline_p_value(self.spline_2, self.density_fraction_2, self.area_2, some_value)
            return [p1, p2]
        else:
            return p1

    def is_similar(self):
        if self.distribution_2 is None:
            return None
        else:
            self.cdf_difference = np.abs(self.scaled_density_fraction_1 - self.scaled_density_fraction_2)
            self.p_similar = full_area(self.cdf_difference, self.step, None, self.include_rate) / 2
            return self.p_similar

    def is_different(self):
        if self.distribution_2 is None:
            return None
        else:
            self.cdf_intersection = np.minimum(self.scaled_density_fraction_1, self.scaled_density_fraction_2)
            self.p_different = full_area(self.cdf_intersection, self.step, None, self.include_rate)
            return self.p_different


if __name__ == '__main__':
    pass
