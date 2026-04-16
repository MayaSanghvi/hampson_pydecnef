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

Preprocessing functions for real-time fMRI volumes using timeseries data.

Each function processes the most recent volume using available timeseries
data from the current run, preparing it for decoding.

Functions:
    1. preproc_to_baseline(): Z-scores using baseline volumes as reference
    2. preproc_to_timeseries(): Z-scores using all volumes so far
    3. preproc_to_model_session(): Z-scores using training session statistics
"""

#############################################################################################
# IMPORT DEPENDENCIES
#############################################################################################

from collections import OrderedDict
from nilearn.signal import clean, _detrend
import numpy as np
from pathlib import Path
import time
from typing import Tuple
import os

#############################################################################################
# FUNCTIONS
#############################################################################################

def preproc_to_baseline(whole_timeseries: np.array, baseline_vols: np.array, preprocessed_dir: str) -> Tuple[np.array, list]:
    """
    Preprocess last coregistered volume using baseline data for z-scoring.

    Steps:
        1 - Detrend baseline (once, then cache)
        2 - Extract last volume from timeseries (detrending currently disabled — see NOTE)
        3 - Z-score using detrended baseline statistics

    Args:
        whole_timeseries: All unpreprocessed coregistered volumes so far (n_vols, n_voxels)
        baseline_vols: All unpreprocessed baseline volumes (n_vols, n_voxels)
        preprocessed_dir: Path to preprocessed volumes folder

    Returns:
        last_zscored_vol: Detrended and z-scored volume (1, n_voxels)
        preproc_time: Processing times for each step (list)
    """
    preproc_time = OrderedDict()
    start_preproc = time.time()

    # Detrend baseline if not already detrended
    start_baseline_detrending = time.time()
    detrended_baseline_path = os.path.join(preprocessed_dir, 'detrended_baseline.npy')
    if Path(detrended_baseline_path).is_file():
        detrended_baseline = np.load(detrended_baseline_path)
    else:
        detrended_baseline = _detrend(
            baseline_vols,
            inplace=False,
            type='linear',
            n_batches=10,
        )
        np.save(detrended_baseline_path, detrended_baseline)
    preproc_time['baseline_detrending'] = time.time() - start_baseline_detrending

    # Extract last volume from timeseries
    # NOTE: Detrending was disabled by DS/NM. The detrend_timeseries function below
    # can be re-enabled by uncommenting the alternative line.
    def detrend_timeseries(whole_timeseries):
        timeseries_to_last_vol = whole_timeseries.copy()
        detrended_timeseries = _detrend(
            timeseries_to_last_vol,
            inplace=False,
            type='linear',
            n_batches=10,
        )
        last_detrended_vol = detrended_timeseries[-1].reshape(1, -1)
        return last_detrended_vol

    start_detrending = time.time()
    # Current behavior: skip detrending, just take the last volume
    last_detrended_vol = whole_timeseries[-1].reshape(1, -1)
    # To re-enable detrending, comment the line above and uncomment this:
    # last_detrended_vol = detrend_timeseries(whole_timeseries)
    preproc_time['detrending_time'] = time.time() - start_detrending

    # Z-score last volume with respect to detrended baseline
    def zscore_func(last_detrended_vol, detrended_baseline):
        zscored_vols = np.vstack([detrended_baseline, last_detrended_vol])
        zscored_vols = clean(
            signals=zscored_vols,
            standardize='zscore',
            detrend=False,
            ensure_finite=True,
        )
        last_zscored_vol = zscored_vols[-1].reshape(1, -1)
        return last_zscored_vol

    start_zscoring = time.time()
    last_zscored_vol = zscore_func(last_detrended_vol, detrended_baseline)
    preproc_time['zscoring_time'] = time.time() - start_zscoring

    # Total preprocessing time
    preproc_time['total_preproc_decoding_time'] = time.time() - start_preproc
    preproc_time = [preproc_time]

    return last_zscored_vol, preproc_time


def preproc_to_timeseries(whole_timeseries: np.array) -> Tuple[np.array, list]:
    """
    Preprocess last coregistered volume using the full timeseries for z-scoring.

    Steps:
        1 - Z-score all volumes using full timeseries (detrending currently disabled)

    Args:
        whole_timeseries: All unpreprocessed coregistered volumes so far (n_vols, n_voxels)

    Returns:
        last_zscored_vol: Z-scored volume (1, n_voxels)
        preproc_time: Processing times for each step (list)
    """
    preproc_time = OrderedDict()
    start_preproc = time.time()

    # Z-score last volume using timeseries to that volume
    start_zscoring = time.time()
    zscored_vols = clean(
        signals=whole_timeseries,
        standardize='zscore',
        detrend=False,  # Detrending disabled (DS/NM). Set to True to re-enable.
        ensure_finite=True,
    )
    last_zscored_vol = zscored_vols[-1].reshape(1, -1)
    preproc_time['zscoring_time'] = time.time() - start_zscoring

    # Total preprocessing time
    preproc_time['total_preproc_decoding_time'] = time.time() - start_preproc
    preproc_time = [preproc_time]

    return last_zscored_vol, preproc_time


def preproc_to_model_session(whole_timeseries: np.array, zscoring_mean: str, zscoring_std: str) -> Tuple[np.array, list]:
    """
    Preprocess last coregistered volume using training session statistics for z-scoring.

    Steps:
        1 - Detrend volume using timeseries data
        2 - Z-score using mean and std from model construction session

    Args:
        whole_timeseries: All unpreprocessed coregistered volumes so far (n_vols, n_voxels)
        zscoring_mean: Path to .npy file containing training session voxel means
        zscoring_std: Path to .npy file containing training session voxel stds

    Returns:
        last_zscored_vol: Detrended and z-scored volume (1, n_voxels)
        preproc_time: Processing times for each step (list)
    """
    preproc_time = OrderedDict()
    start_preproc = time.time()

    # Load z-scoring reference data
    zscoring_mean = np.load(zscoring_mean)
    zscoring_std = np.load(zscoring_std)

    # Detrend timeseries to last volume
    def detrend_func(whole_timeseries):
        timeseries_including_last_vol = whole_timeseries.copy()
        detrended_timeseries = _detrend(
            timeseries_including_last_vol,
            inplace=False,
            type='linear',
            n_batches=10,
        )
        detrended_last_vol = detrended_timeseries[-1].reshape(1, -1)
        return detrended_last_vol

    start_detrending = time.time()
    detrended_last_vol = detrend_func(whole_timeseries)
    preproc_time['detrending_time'] = time.time() - start_detrending

    # Z-score using training session statistics
    def zscore_func(detrended_last_vol, zscoring_mean, zscoring_std):
        last_zscored_vol = (detrended_last_vol - zscoring_mean) / zscoring_std
        return last_zscored_vol

    start_zscoring = time.time()
    last_zscored_vol = zscore_func(detrended_last_vol, zscoring_mean, zscoring_std)
    preproc_time['zscoring_time'] = time.time() - start_zscoring

    # Total preprocessing time
    preproc_time['total_preproc_time'] = time.time() - start_preproc
    preproc_time = [preproc_time]

    return last_zscored_vol, preproc_time