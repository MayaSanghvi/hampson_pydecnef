#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converts DICOM files to NIfTI format and performs:
    - Deoblique correction
    - Brain extraction using AFNI Automask
    - Spatial registration against a reference volume (AFNI 3dvolreg)

All processing is done in the provided directory structure.
After processing each volume, intermediate files are removed to save space.

Compatible with Siemens DICOM naming conventions (e.g., MRe.1.3.12.2...) by
using pydicom to read InstanceNumber for proper volume ordering.
"""

from pathlib import Path
from nipype.interfaces import afni as afni
import subprocess
import os
import pydicom

# Define the file structure
exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir))
raw_vols_dir = Path(os.path.join(exp_dir, '2_data', 'raw', 'func'))
preprocessed_dir = os.path.join(exp_dir, '2_data', 'preprocessed')
func_dir = os.path.join(preprocessed_dir, 'func')
os.makedirs(func_dir, exist_ok=True)
example_func_dir = os.path.join(preprocessed_dir, 'example_func')
example_func = os.path.join(example_func_dir, 'example_func_deoblique_brain.nii')


def get_instance_number(filepath):
    """Read DICOM InstanceNumber for proper volume ordering."""
    try:
        dcm = pydicom.dcmread(str(filepath), stop_before_pixels=True)
        return int(dcm.InstanceNumber)
    except Exception:
        return 0


for folder in sorted(raw_vols_dir.iterdir()):
    if not folder.is_dir() or folder.name.startswith('.'):
        continue

    # Create a subdirectory for the current run's data
    run_dir = Path(os.path.join(func_dir, folder.stem))
    os.makedirs(run_dir, exist_ok=True)

    # Get all DICOM files (no extension filter — works with Siemens naming)
    vol_files = [
        f for f in folder.iterdir()
        if f.is_file() and not f.name.startswith('.')
    ]

    # Sort by DICOM InstanceNumber for correct volume order
    vol_files.sort(key=get_instance_number)

    print(f"Processing {len(vol_files)} volumes in {folder.name}...")

    for vol_idx, vol_file in enumerate(vol_files, 1):
        # Use clean indexed name to avoid Siemens filename issues
        vol_name = f'vol_{vol_idx:04d}'

        # Convert DICOM to NIfTI using dcm2niix
        subprocess.run(
            [f'dcm2niix -z n -f {vol_name} -o {run_dir} -s y {vol_file}'],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        nifti_file = os.path.join(run_dir, vol_name + '.nii')

        # Deoblique correction
        deoblique = afni.Warp()
        deoblique.inputs.in_file = nifti_file
        deoblique.inputs.deoblique = True
        deoblique.inputs.gridset = example_func
        deoblique.inputs.num_threads = 4
        deoblique.inputs.outputtype = 'NIFTI'
        deoblique_file = os.path.join(run_dir, vol_name + '_deoblique.nii')
        deoblique.inputs.out_file = str(deoblique_file)
        deoblique.run()

        # Brain extraction
        brainextraction = afni.Automask()
        brainextraction.inputs.in_file = deoblique_file
        brainextraction.inputs.erode = 1
        brainextraction.inputs.clfrac = 0.5
        brainextraction.inputs.num_threads = 4
        brainextraction.inputs.outputtype = 'NIFTI'
        brain_file = os.path.join(run_dir, vol_name + '_deoblique_brain.nii')
        brainmask_file = os.path.join(run_dir, vol_name + '_deoblique_brainmask.nii')
        brainextraction.inputs.brain_file = brain_file
        brainextraction.inputs.out_file = brainmask_file
        brainextraction.run()

        # Spatial registration to reference volume
        volreg = afni.Volreg()
        volreg.inputs.in_file = brain_file
        volreg.inputs.basefile = example_func
        volreg.inputs.args = '-heptic'
        volreg.inputs.num_threads = 4
        volreg.inputs.outputtype = 'NIFTI'
        oned_file = os.path.join(run_dir, vol_name + '_deoblique_brain_corregister.1D')
        oned_matrix_file = os.path.join(run_dir, vol_name + '_deoblique_brain_corregister.aff12.1D')
        md1d_file = os.path.join(run_dir, vol_name + '_deoblique_brain_corregister_md.1D')
        corregister_file = os.path.join(run_dir, vol_name + '_deoblique_brain_corregister.nii')
        volreg.inputs.oned_file = oned_file
        volreg.inputs.oned_matrix_save = oned_matrix_file
        volreg.inputs.md1d_file = md1d_file
        volreg.inputs.out_file = corregister_file
        volreg.run()

        # Clean up intermediate files, keep only coregistered volumes
        for file in run_dir.glob('*'):
            if 'corregister.nii' not in str(file):
                file.unlink()

        print(f"  Processed volume {vol_idx}/{len(vol_files)}")