#!/usr/bin/env python
"""
Predict Glucose from PPG Windows CSV

Usage:
    python predict_glucose.py <filtered_windows.csv>

Example:
    python predict_glucose.py web_app_data/case_2_SNUADC_PLETH/case_2_SNUADC_PLETH_raw_cleansed_filtered_windows_detailed.csv
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference.glucose_from_csv import main

if __name__ == '__main__':
    main()
