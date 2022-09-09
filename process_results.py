#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import numpy as np
import shutil

levels_dir = 'levels'


def collect_results():
    """
    Copy all outputted water surface data files from their individual case folders
    to the levels folder
    """
    print("Collecting Results...")
    os.makedirs(levels_dir, exist_ok=True)
    for run in os.listdir('output'):
        i = int(run.lstrip('output_'))
        try:
            shutil.copyfile(os.path.join('output', run, 'Surface_ElevationPos.csv'),
                            os.path.join(levels_dir, f'Surface_ElevationPos_{i:04}.csv'))
        except FileNotFoundError:
            print(f"Failed {run}")


def convert_results_to_numpy():
    """
    Parse and normalize the text water surface data
    """
    # Load and clean the case definitions from Cases.csv
    # Strips useless pieces and convert to float
    with open('Cases.csv', 'r') as f:
        data = [tuple(float(e.strip()) for e in line.split(',')[:4]) for line in f.readlines()[1:]]

    # Collect and normalize the water surface data
    configs = []
    res = []
    for f in sorted(os.listdir('levels')):
        # Find the case id from the file name
        id = int(f.rstrip('.csv')[-3:])
        print(f"Processing {id}")

        # Log the corresponding parameter values
        configs.append(data[id - 1])

        # Load the water surface data
        df = pd.read_csv(os.path.join('levels', f), sep=";", header=0, index_col=[0, 1])

        # Compute the standard deviation data that we are after
        std = df.std(axis=1)

        # Compute the inverse rotation matrix to center and align our frame of reference on the buoy row
        buoy_angle = data[id - 1][3]
        theta = np.radians(buoy_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c))).T
        R_inv = np.linalg.inv(R)

        # Unstack the standard deviation data
        std.index = pd.MultiIndex.from_tuples(tuple(map(tuple, (np.array(list(std.index)) @ R_inv).round())))
        res.append(std.unstack().sort_index(ascending=True).to_numpy())
    configs = np.array(configs)
    labels = np.stack(res)
    np.save('data.npy', {"configs": configs, "labels": labels})


if __name__ == '__main__':
    collect_results()
    convert_results_to_numpy()
