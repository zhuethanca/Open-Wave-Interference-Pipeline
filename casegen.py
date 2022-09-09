#!/usr/bin/env python
# coding: utf-8


import numpy as np
import itertools
from util import load_config


def casegen():
    """
    Generate case definitions to Cases.csv

    Two types of case definitions can be generated
    1. All combinations of specified values for the parameters (grid)
    2. Random values within specified ranges (random)

    """

    config = load_config()['Case Generation']
    grid_settings = config['Grid Settings']
    random_settings = config['Random Settings']

    height = grid_settings['Wave Height Samples (m)']
    period = grid_settings['Wave Period Samples (s/cycle)']
    distance = grid_settings['Bouy Distance Samples (m)']
    angle = grid_settings['Wave Angle Samples (°)']

    ranges = np.array([
        random_settings['Wave Height Range (m)'],
        random_settings['Wave Period Range (s/cycle)'],
        random_settings['Bouy Distance Range (m)'],
        random_settings['Wave Angle Range (°)'],
    ])

    n = random_settings['Count']
    random_cases = (
            np.random.rand(n, 4)  # Uniform Random [0-1] values in the shape (n, 4)
            * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]  # Scale random values to [min, max]
    ).round(2)

    with open('Cases.csv', 'w') as f:
        f.write('Wave Height, Wave Period, Buoy Distance, Buoy Angle, Baseline, Complete\n')
        # Write the cartesian product of all specified grid values
        for d, a, p, h in itertools.product(distance, angle, period, height):
            f.write(f"{h}, {p}, {d}, {a}, False, False\n")

        # Write the random cases
        for row in random_cases:
            f.write(", ".join(str(r) for r in row) + ", False, False\n")


if __name__ == '__main__':
    casegen()
