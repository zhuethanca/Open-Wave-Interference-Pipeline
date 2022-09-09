#!/usr/bin/env python
# coding: utf-8


import os
import json
import platform
from typing import Tuple

import numpy as np
import subprocess
import itertools
import csv
import re
import ast
import shutil

from util import load_config

raw_exes = {
    "GenCase": "GenCase",
    "DualSPHysics": "DualSPHysics5.0",
    "MeasureTool": "MeasureTool",
    "BoundaryVTK": "BoundaryVTK",
    "IsoSurface": "IsoSurface",
}


def setup() -> Tuple[dict, dict, list]:
    """
    Setup environment based on platform.

    Import config and cases data.
    """
    # Generate executable names based on platform
    if platform.system() == 'Windows':
        exes = dict((k, v + "_win64.exe") for k, v in raw_exes.items())
    elif platform.system() == 'Linux':
        os.environ["LD_LIBRARY_PATH"] = os.path.join(os.getcwd(), 'lib') + ":" + os.environ["LD_LIBRARY_PATH"]
        exes = dict((k, v + "_linux64") for k, v in raw_exes.items())
    else:
        raise OSError(f"Platform {platform.system()} not supported!")

    # Load Config
    config = load_config()['Simulation']

    # Load Cases
    with open('Cases.csv', newline='') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip Header Row
        # Strip and evaluate literals in case data
        clean_cases = (list(map(ast.literal_eval, map(str.strip, line))) for line in reader)
        # Slice the specified case range
        cases = list(itertools.islice(clean_cases, config['Case Start'] - 1, config['Case End'] - 1))
    return config, exes, cases


def run_case(
        case_id: int,
        wave_height: float, wave_period: float,
        buoy_distance: float, buoy_angle: float,
        baseline: bool, complete: bool,
        config: dict, exes: dict
):
    """
    Run a single case.

    :param int case_id: The numerical identifier for the case
    :param float wave_height: The height of the wave
    :param float wave_period: The period of the wave
    :param float buoy_distance: The distance between a edge buoy to the center buoy
    :param float buoy_angle: The angle of the buoy row to the incoming wave
    :param bool baseline: Whether or not to run a baseline case
    :param bool complete: Whether the case is complete
    :param dict config: Config Data
    :param dict exes: Dictionary of all executable names
    """
    if complete:
        print(f"Case {case_id} already complete, skipping...")
        return
    else:
        print(f"Running {case_id}...")

    # Load the case definition template
    # Changes to the baseline definition if necessary
    with open(os.path.join('casedata',
                           'simulation.casedata_template_baseline.xml'
                           if baseline else
                           'simulation.casedata_template.xml')) as f:
        template = f.read()

    # Load the center buoy location
    buoy_center = config['Buoy Center']
    center = np.array([[buoy_center[0], buoy_center[1]]])
    # 3D center
    center3 = np.array([[buoy_center[0], buoy_center[1], 0]])

    # Position of the 3 buoys in coordinates centered on the center buoy
    buoy_pos = np.array([
        [0, -buoy_distance],
        [0, 0],
        [0, +buoy_distance],
    ])

    # Rotation Matrix
    theta = np.radians(buoy_angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c))).T
    R3 = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))).T

    buoy_pos = buoy_pos @ R + center

    def cartesian_product(*arrays):
        ndim = len(arrays)
        return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)

    # Generate Grid Points for wave height sampling
    x_steps = np.arange(0, config['Sample Grid Length'] + 1e-4, config['Sample Grid Step'], dtype=np.float) \
              - config['Sample Grid Neg X Offset']
    y_steps = np.arange(-config['Sample Grid Width'] / 2, config['Sample Grid Width'] / 2 + 1e-4,
                        config['Sample Grid Step'], dtype=np.float)
    z_steps = np.arange(0, config['Sample Grid Height'] + 1e-4, config['Sample Grid VStep'], dtype=np.float)

    grid_points = cartesian_product(x_steps, y_steps, z_steps) @ R3 + center3

    # Prepare parameters for filling in template
    params = {
        "sim_dur": config['Simulation Duration'],
        "ptc_dst": config['Interparticle Distance'],
        "buoy_top_z": config['Buoy Top Z'],
        "buoy_bot_z": config['Buoy Bot Z'],
        "buoy_attach_z": config['Buoy Bot Z'] + 0.1,
        'damping': config['Damping'],
        'wave_height': wave_height,
        'wave_period': wave_period,
        'buoy_1_x': buoy_pos[0, 0],
        'buoy_1_y': buoy_pos[0, 1],
        'buoy_2_x': buoy_pos[1, 0],
        'buoy_2_y': buoy_pos[1, 1],
        'buoy_3_x': buoy_pos[2, 0],
        'buoy_3_y': buoy_pos[2, 1],
    }

    with open(os.path.join('casedata', 'simulation.casedata_Def.xml'), 'w') as f:
        f.write(template.format(**params))

    bin_dir = os.path.join(os.getcwd(), 'bin')
    def_dir = os.path.join(os.getcwd(), 'casedata')
    def_xml = os.path.join(def_dir, 'simulation.casedata_Def')

    case_dir = os.path.join(os.getcwd(), 'cases', f'case_{case_id}')
    os.makedirs(case_dir, exist_ok=True)

    case_file = os.path.join(case_dir, 'simulation.casedata')

    output_dir = os.path.join(os.getcwd(), 'output', f'output_{case_id}', "")
    os.makedirs(output_dir, exist_ok=True)

    # Write Grid Points for wave height sampling
    points_file = os.path.join(output_dir, 'points.txt')
    with open(points_file, 'w') as f:
        f.write('POINTS #' + os.linesep)
        for x, y, z in grid_points:
            f.write(" ".join((str(x), str(y), str(z))) + os.linesep)

    # Copy over buoy models
    for f in os.listdir(def_dir):
        if f.endswith('.obj'):
            shutil.copy(os.path.join(def_dir, f), case_dir)

    # Attempt to find previous progress
    try:
        if config["Allow Continuation After Fail"]:
            restart_part = max(
                part for part in os.listdir(output_dir) if part.startswith('Part_') and part.endswith('.bi4')).lstrip(
                "Part_").rstrip(".bi4")
        else:
            restart_part = None
    except:
        restart_part = None

    if restart_part is not None:
        print(f"Previous Run Found, Continuing from {restart_part}")
    else:
        print("Skipping Generate Case...")
    try:
        # Run Gen Case tool. Generates case folder from template.
        if config["Generate Case"] and restart_part is None:
            subprocess.run([os.path.join(bin_dir, exes['GenCase']), def_xml, case_file, '-save:+all'], cwd=def_dir,
                           check=True)

        # Run simulation
        if config["Run Case"]:
            subprocess.run([os.path.join(bin_dir, exes['DualSPHysics']), case_file, output_dir, '-gpu', '-svres'] + (
                [] if restart_part is None else [
                    "-partbegin:" + restart_part,
                    output_dir,
                ]), cwd=case_dir, check=True)

        # Reconstruct surface for visualization
        if config["Generate Surface"]:
            subprocess.run([os.path.join(bin_dir, exes['IsoSurface']),
                            '-dirin', output_dir,
                            '-saveiso', os.path.join(output_dir, 'FileFluid.vtk'),
                            '-onlytype:+fluid'], cwd=case_dir, check=True)
            subprocess.run([os.path.join(bin_dir, exes['BoundaryVTK']),
                            '-loadvtk', 'AutoActual',
                            '-motiondata', output_dir,
                            '-onlymk:11-16',
                            '-savevtkdata', os.path.join(output_dir, 'FileBuoy.vtk')], cwd=case_dir, check=True)

        # Export water surface height data
        if config["Generate Elevation"]:
            subprocess.run([os.path.join(bin_dir, exes['MeasureTool']),
                            '-dirin', output_dir,
                            '-points', os.path.join(output_dir, 'points.txt'),
                            '-savecsv', os.path.join(output_dir, 'Surface.csv'),
                            '-height',
                            '-onlytype:+fluid',
                            '-elevationoutput:all'], cwd=case_dir, check=True)
    finally:
        # Clean up simulation data files
        if config["Cull Output"]:
            shutil.rmtree(case_dir)
            outputs = os.listdir(output_dir)
            for f in outputs:
                if re.match(r'Part_\d+.bi4', f):
                    os.remove(os.path.join(output_dir, f))
            os.remove(os.path.join(output_dir, 'points.txt'))
    # Update cases with finished marker
    with open('Cases.csv', 'r') as f:
        lines = f.readlines()
        lines[case_id] = ','.join(lines[case_id].split(',')[:-1] + [" True"]) + "\n"
    with open('Cases.csv', 'w') as f:
        f.write(''.join(lines))


def main():
    config, exes, cases = setup()
    for case_id in range(config['Case Start'], config['Case End']):
        try:
            run_case(case_id, *cases[case_id - config['Case Start']])
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
