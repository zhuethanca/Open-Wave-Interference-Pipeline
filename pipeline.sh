#!/bin/bash

echo "Generating Cases"
python casegen.py
echo "Running Simulations"
python run_simulations.py
echo "Processing Results"
python process_results.py
echo "Fitting Models"
python fit.py