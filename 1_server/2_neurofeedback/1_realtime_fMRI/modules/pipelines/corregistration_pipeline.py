############################################################################
# AUTHORS: Pedro Margolles & David Soto
# EMAIL: pmargolles@bcbl.eu, dsoto@bcbl.eu
# COPYRIGHT: Copyright (C) 2021-2022, pyDecNef
# URL: https://pedromargolles.github.io/pyDecNef/
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
"""
Description:

Real-time fMRI volume coregistration for neuroimaging analysis.

Processes DICOM files by converting them to NIfTI format, deobliques the images,
extracts the brain, and aligns the volume with a reference using AFNI tools.
The processed volume is masked with an ROI mask and saved for further processing.

Steps:
    1. Copy DICOM file to preprocessing directory
    2. Convert DICOM to NIfTI format
    3. Deoblique NIfTI file to match orientation
    4. Extract brain tissue using AFNI Automask
    5. Align extracted brain to reference volume using Volreg
    6. Mask aligned volume with ROI mask

Inputs:
    - vol_file: Path to raw fMRI DICOM file
    - vol_idx: Volume index (used for clean output filenames)
    - mask_file: Path to Region of Interest (ROI) mask in NIfTI format
    - ref_vol_file: Path to reference volume in NIfTI format
    - preprocessed_dir: Path for saving processed files

Outputs:
    - preproc_vol: Preprocessed and aligned fMRI volume (numpy array)
    - corregistration_time: Dictionary with processing times for each step
"""

#############################################################################################
# IMPORT DEPENDENCIES
#############################################################################################

from collections import OrderedDict
from nipype.interfaces import afni as afni
from nilearn.masking import apply_mask
import numpy as np
from pathlib import Path
from shutil import copyfile
import subprocess
import time
from typing import Tuple

#############################################################################################
# FUNCTIONS
#############################################################################################

def corregister_vol(vol_file: str, vol_idx: int, mask_file: str, ref_vol_file: str, preprocessed_dir: str) -> Tuple[np.array, list]:
    """
    Coregistration pipeline to be applied to raw DICOM volumes.

    Args:
        vol_file: raw fMRI volume file path (DICOM file)
        vol_idx: volume index number, used to generate clean output filenames
                 (avoids issues with Siemens DICOM naming conventions)
        mask_file: R.O.I. mask to apply to vol_file (NIfTI file, uncompressed)
        ref_vol_file: reference volume for coregistration (NIfTI file, uncompressed)
        preprocessed_dir: path where preprocessed files will be saved

    Returns:
        preproc_vol: coregistered and masked volume (numpy array)
        corregistration_time: processing time for each pipeline step (list)
    """

    corregistration_time = OrderedDict()
    start_corregistration = time.time()

    # Set working files
    vol_file = Path(vol_file)
    mask_file = Path(mask_file)
    ref_vol_file = Path(ref_vol_file)
    preprocessed_dir = Path(preprocessed_dir)

    # Use vol_idx for clean filenames (avoids Siemens DICOM naming issues)
    vol_name = f'vol_{vol_idx:04d}'

    # Copy DICOM file to preprocessed_dir
    start_dcm_copy = time.time()
    dcm_copy = preprocessed_dir / (vol_name + '.dcm')
    copyfile(vol_file, dcm_copy)
    corregistration_time['dcm_copy_time'] = time.time() - start_dcm_copy

    # Load DICOM, convert to NIfTI format, and store uncompressed NIfTI in preprocessed_dir
    start_nifti_conver = time.time()
    subprocess.run(
        [f'dcm2niix -z n -f {vol_name} -o {preprocessed_dir} -s y {dcm_copy}'],
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    nifti_file = preprocessed_dir / (vol_name + '.nii')
    corregistration_time['nifti_conver_time'] = time.time() - start_nifti_conver

    # Deoblique uncompressed NIfTI file to match cardinal coordinates orientation
    start_deoblique = time.time()
    deoblique = afni.Warp()
    deoblique.terminal_output = 'file_split'
    deoblique.inputs.in_file = nifti_file
    deoblique.inputs.deoblique = True
    deoblique.inputs.gridset = ref_vol_file
    deoblique.inputs.num_threads = 4
    deoblique.inputs.outputtype = 'NIFTI'
    deoblique_file = preprocessed_dir / (vol_name + '_deoblique.nii')
    deoblique.inputs.out_file = deoblique_file
    deoblique.run()
    corregistration_time['deoblique_time'] = time.time() - start_deoblique

    # Brain extraction to improve coregistration of functional data
    start_brainextraction = time.time()
    brainextraction = afni.Automask()
    brainextraction.terminal_output = 'file_split'
    brainextraction.inputs.in_file = deoblique_file
    brainextraction.inputs.erode = 1
    brainextraction.inputs.clfrac = 0.5
    brainextraction.inputs.num_threads = 4
    brainextraction.inputs.outputtype = 'NIFTI'
    brain_file = preprocessed_dir / (vol_name + '_deoblique_brain.nii')
    brainmask_file = preprocessed_dir / (vol_name + '_deoblique_brainmask.nii')
    brainextraction.inputs.brain_file = brain_file
    brainextraction.inputs.out_file = brainmask_file
    brainextraction.run()
    corregistration_time['brainextraction_time'] = time.time() - start_brainextraction

    # Coregister this extracted brain to reference volume
    start_volreg = time.time()
    volreg = afni.Volreg()
    volreg.terminal_output = 'file_split'
    volreg.inputs.in_file = brain_file
    volreg.inputs.basefile = ref_vol_file
    volreg.inputs.args = '-heptic'
    volreg.inputs.num_threads = 4
    volreg.inputs.outputtype = 'NIFTI'
    oned_file = preprocessed_dir / (vol_name + '_deoblique_brain_corregister.1D')
    oned_matrix_file = preprocessed_dir / (vol_name + '_deoblique_brain_corregister.aff12.1D')
    md1d_file = preprocessed_dir / (vol_name + '_deoblique_brain_corregister_md.1D')
    corregister_file = preprocessed_dir / (vol_name + '_deoblique_brain_corregister.nii')
    volreg.inputs.oned_file = oned_file
    volreg.inputs.oned_matrix_save = oned_matrix_file
    volreg.inputs.md1d_file = md1d_file
    volreg.inputs.out_file = corregister_file
    volreg.run()
    corregistration_time['volreg_time'] = time.time() - start_volreg

    # Mask with ROI
    start_mask = time.time()
    mask_file = str(mask_file)
    corregister_file = str(corregister_file)
    preproc_vol = apply_mask(
        imgs=corregister_file,
        mask_img=mask_file,
        smoothing_fwhm=None,
        ensure_finite=True,
    )
    corregistration_time['mask_time'] = time.time() - start_mask

    # Total coregistration time
    corregistration_time['total_corregistration_time'] = time.time() - start_corregistration
    corregistration_time = [corregistration_time]

    return preproc_vol, corregistration_time