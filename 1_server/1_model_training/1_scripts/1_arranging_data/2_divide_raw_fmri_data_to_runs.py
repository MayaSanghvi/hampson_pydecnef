############################################################################
# AUTHORS: Najemeddine Abdennour
# EMAIL: nabdennour@bcbl.eu
# COPYRIGHT: Copyright (C) 2024-2025, pyDecNef
# URL: https://github.com/najemabdennour/pyDecNef
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
"""
Organizes raw fMRI data into separate runs within the decoder training directory.

Uses pydicom to read SeriesNumber from DICOM headers to determine which run each
file belongs to. This approach is compatible with any scanner's DICOM naming
convention, including Siemens long UID filenames (e.g., MRe.1.3.12.2...).

Steps:
    1. Determine the folder containing raw DICOM data
    2. Read DICOM headers to extract SeriesNumber for each file
    3. Group files by SeriesNumber (each series = one run)
    4. Copy files into run-specific subdirectories under raw/func/
"""

import os
from datetime import datetime
from collections import defaultdict
import shutil
import pydicom

# Path where scanner exports DICOMs via Samba
recorded_data_main_folder = "/data/realtime_dicoms/"  # Update to your scanner's export path

# Toggle for test mode to use local data
test_mode = True
if test_mode:
    recorded_data_main_folder = os.path.abspath(
        os.path.join(os.getcwd(), os.pardir, os.pardir, "2_data", "recorded_data")
    )

print("Main folder for recorded data:", recorded_data_main_folder)

# Get list of subdirectories in the main folder
all_items = os.listdir(recorded_data_main_folder)
data_folders = [
    i for i in all_items
    if os.path.isdir(os.path.join(recorded_data_main_folder, i)) and not i.startswith(".")
]

if data_folders:
    if test_mode:
        recorded_directories_list_today = data_folders
    else:
        # Filter directories by today's date
        today_date_str = datetime.today().strftime("%Y%m%d")
        recorded_directories_list_today = [
            i for i in data_folders if i.startswith(today_date_str)
        ]

    # Convert to full paths and sort by modification time
    recorded_data_directories_list_paths = [
        os.path.join(recorded_data_main_folder, i)
        for i in recorded_directories_list_today
    ]
    recorded_data_directories_list_paths.sort(key=lambda x: os.path.getmtime(x))

    # Use the most recent folder
    try:
        raw_training_data_path = recorded_data_directories_list_paths[-1]
    except IndexError:
        print(f"No new folders created today. Using fallback path: {recorded_data_main_folder}")
        raw_training_data_path = recorded_data_main_folder
else:
    raw_training_data_path = recorded_data_main_folder

print(f"Chosen raw training path to extract runs from: {raw_training_data_path}")

# Get all non-hidden files
raw_data_list = [
    file for file in os.listdir(raw_training_data_path) if not file.startswith(".")
]

print(f"Found {len(raw_data_list)} files. Reading DICOM headers to identify runs...")

# Group files by SeriesNumber from DICOM headers
runs = defaultdict(list)
skipped = 0

for file in raw_data_list:
    file_path = os.path.join(raw_training_data_path, file)
    try:
        dcm = pydicom.dcmread(file_path, stop_before_pixels=True)  # only read header, skip pixel data for speed
        series_number = int(dcm.SeriesNumber)
        runs[series_number].append(file)
    except Exception as e:
        skipped += 1
        print(f"  Skipping {file}: {e}")

if skipped > 0:
    print(f"Skipped {skipped} files that could not be read as DICOM.")

# Sort the series numbers so runs are in order
sorted_series = sorted(runs.keys())

print(f"Found {len(sorted_series)} runs (SeriesNumbers: {sorted_series})")
for series_num in sorted_series:
    print(f"  Series {series_num}: {len(runs[series_num])} files")

# Path to decoder training data folder
decoder_training_folder_path = os.path.abspath(
    os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir, "2_data")
)
print(f"Decoder training data folder path: {decoder_training_folder_path}")

# Organize files into run subdirectories
all_copied = True
for i, series_num in enumerate(sorted_series):
    dest_path = os.path.join(decoder_training_folder_path, "raw", "func", f"run_{i}")

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
        print(f"Copying {len(runs[series_num])} files for Series {series_num} -> run_{i}")
        for file in runs[series_num]:
            shutil.copy(
                src=os.path.join(raw_training_data_path, file),
                dst=os.path.join(dest_path, file),
            )
    else:
        print(f"The run_{i} folder already exists. No action taken.")
        all_copied = False

if all_copied:
    print("fMRI data successfully organized into run folders.")