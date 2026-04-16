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

Trial-level fMRI decoding for real-time neurofeedback.

Processes preprocessed volumes through a pretrained classifier to produce
decoding probabilities for the target class. Supports three decoding modes:
    1. average_hrf_peak_vols: Average volumes then decode once
    2. average_probs: Decode each volume then average probabilities
    3. dynamic: Decode each volume independently for immediate feedback

This script performs trial-level fMRI data decoding for neuroimaging analysis, 
specifically designed for real-time applications. It processes preprocessed 
neuroimaging volumes (e.g., fMRI) through multiple decoding approaches to 
classify or predict cognitive states, brain states, or any target 
conditions of interest. The script supports various machine learning 
algorithms and provides flexible processing options based on trial 
characteristics.

Key Features:
    - Supports real-time trial-level fMRI data decoding
    - Implements dynamic decoding for immediate feedback in experimental setups
    - Allows for both single-trial and aggregated (HRF-peak) decoding approaches
    - Compatible with multiple machine learning algorithms (e.g., Logistic Regression, SVM)
    - Includes time-efficient processing to handle rapid data streams
    - Provides detailed timing information for each decoding step


Inputs:
    - preproc_vols_data: List of masked and preprocessed fMRI volumes (NifTI format)
    - model_file: Path to a pretrained machine learning model (e.g., Logistic Regression, SVM)
    - ground_truth: Target class label corresponding to the ground truth condition
   (integer value)

Outputs:
    - decoding_prob: Probability of correct classification for the given volume/condition
    - decoding_time: List containing timing information for each decoding step
    - vols_decoding_probs: Optional output with individual probabilities for each HRF peak volume

Performance Considerations:

The script is optimized for efficient processing, ensuring minimal latency 
between data acquisition and model evaluation. It supports both single-trial 
dynamic decoding and aggregated decoding approaches, allowing flexibility in 
experimental design. The use of pre-trained models and efficient prediction 
algorithms ensures that even high-resolution neuroimaging data can be processed 
quickly.
"""

#############################################################################################
# IMPORT DEPENDENCIES
#############################################################################################

from collections import OrderedDict
from joblib import load
import numpy as np
import time
from typing import Tuple

#############################################################################################
# FUNCTIONS
#############################################################################################

def average_hrf_peak_vols_decoding(preproc_vols_data: list, model_file: str, ground_truth: int) -> Tuple[np.array, list]:
    """
    Average volumes within a trial HRF peak before decoding a single averaged volume.

    Steps:

        1 - Load model
        2 - Average volumes of interest (i.e., volumes within HRF peak) onto a single volume
        3 - Predict class probabilities over this trial averaged volume
        4 - Get probability of decoding ground truth class
    
    Args:
        preproc_vols_data: masked and preprocessed volumes (list of arrays, each 1 x n_voxels)
        model_file: path to pretrained scikit-learn model
        ground_truth: target class index in model probability predictions

    Returns:
        decoding_prob: probability of decoding ground truth class (float)
        decoding_time: processing times for each step (list)
    """
    decoding_time = OrderedDict()

    # Load pretrained model
    clf_fit = load(model_file)

    start_decoding = time.time()

    # Average volumes within HRF peak
    start_average_vols = time.time()
    average_preproc = np.average(preproc_vols_data, axis=0)
    average_preproc = average_preproc.reshape(1, -1)
    decoding_time['average_vols_time'] = time.time() - start_average_vols

    # Predict class probability
    start_prediction = time.time()
    class_probabilities = clf_fit.predict_proba(average_preproc)
    decoding_prob = class_probabilities[0][ground_truth]
    print("THE DECODING PROBABILITY:", decoding_prob)
    decoding_time['prediction_time'] = time.time() - start_prediction

    # Total decoding time
    decoding_time['total_decoding_time'] = time.time() - start_decoding
    decoding_time = [decoding_time]

    return decoding_prob, decoding_time


def average_probs_decoding(preproc_vols_data: list, model_file: str, ground_truth: int) -> Tuple[np.array, list, list]:
    """
    Decode each HRF peak volume independently then average their probabilities.

    Steps:

        1 - Load model
        2 - Predict class probabilities for each HRF peak volume
        3 - Get probability of decoding ground truth class for each HRF peak volume
        4 - Average probabilities of decoding ground truth across volumes

    Args:
        preproc_vols_data: masked and preprocessed volumes (list of arrays, each 1 x n_voxels)
        model_file: path to pretrained scikit-learn model
        ground_truth: target class index in model probability predictions

    Returns:
        averaged_decoding_prob: mean decoding probability across HRF volumes (float)
        vols_decoding_probs: per-volume decoding probabilities (list of floats)
        decoding_time: processing times for each step (list)
    """
    decoding_time = OrderedDict()

    # Load pretrained model
    clf_fit = load(model_file)

    start_decoding = time.time()
    start_decoding_vols = time.time()
    vols_decoding_probs = []

    # Decode each volume within HRF peak
    for vol in preproc_vols_data:
        class_probabilities = clf_fit.predict_proba(vol)
        decoding_prob = class_probabilities[0][int(ground_truth)]
        vols_decoding_probs.append(decoding_prob)

    print("THE TOTAL DECODING PROBABILITIES FOR THE VOLUMES:", vols_decoding_probs)
    averaged_decoding_prob = np.average(vols_decoding_probs)
    decoding_time['decoding_vols_time'] = time.time() - start_decoding_vols

    # Total decoding time
    decoding_time['total_decoding_time'] = time.time() - start_decoding
    decoding_time = [decoding_time]

    return averaged_decoding_prob, vols_decoding_probs, decoding_time


def dynamic_decoding(preproc_vol: np.array, model_file: str, ground_truth: int) -> Tuple[np.array, list]:
    """
    Decode a single volume independently for real-time dynamic feedback.

    Args:
        preproc_vol: a single masked and preprocessed volume (1 x n_voxels)
        model_file: path to pretrained scikit-learn model
        ground_truth: target class index in model probability predictions

    Returns:
        decoding_prob: probability of decoding ground truth class (float)
        decoding_time: processing times for each step (list)
    """
    decoding_time = OrderedDict()

    # Load pretrained model
    clf_fit = load(model_file)

    start_decoding = time.time()

    # Predict class probability
    start_prediction = time.time()
    class_probabilities = clf_fit.predict_proba(preproc_vol)
    decoding_prob = class_probabilities[0][ground_truth]
    print("THE DECODING PROBABILITY:", decoding_prob)
    decoding_time['prediction_time'] = time.time() - start_prediction

    # Total decoding time
    decoding_time['total_decoding_time'] = time.time() - start_decoding
    decoding_time = [decoding_time]

    return decoding_prob, decoding_time