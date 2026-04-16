############################################################################
# AUTHORS: Pedro Margolles & David Soto
# EMAIL: pmargolles@bcbl.eu, dsoto@bcbl.eu
# COPYRIGHT: Copyright (C) 2021-2022, pyDecNef
# URL: https://pedromargolles.github.io/pyDecNef/
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
"""
Simulates a MRI scanner by copying DICOM files from 'real_data' to 'outputs'
at TR intervals. Used to test the neurofeedback pipeline without a real scanner.
"""

import shutil
import time
from pathlib import Path
import os
from colorama import init, Fore
init()

#############################################################################################
# fMRI SIMULATION VARIABLES
#############################################################################################

TR = 2

#############################################################################################
# DIRECTORIES & DATA
#############################################################################################

script_dir = Path(__file__).absolute().parent

outputs_dir = Path(os.path.join(script_dir, 'outputs'))

# Remove all files in outputs_dir
if outputs_dir.exists():
    for file in outputs_dir.iterdir():
        if file.is_file():
            file.unlink()

# Folder with fMRI DICOM files
real_data = Path(os.path.join(script_dir, 'real_data'))

#############################################################################################
# DATA TRANSFER FROM REAL_DATA FOLDER TO OUTPUTS FOLDER
#############################################################################################

volumes = sorted(list(real_data.glob('*')))
print(Fore.YELLOW + f'Found {len(volumes)} volumes in real_data folder.\n')

for i, volume in enumerate(volumes, 1):
    print(Fore.YELLOW + f'\n[PROCESSING] Generating volume {i}/{len(volumes)}: {volume.name}...')
    time.sleep(TR)
    shutil.copy(str(volume), str(outputs_dir))
    print(Fore.GREEN + '[OK]')