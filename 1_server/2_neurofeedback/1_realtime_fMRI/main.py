############################################################################
# AUTHORS: Pedro Margolles & David Soto
# EMAIL: pmargolles@bcbl.eu, dsoto@bcbl.eu
# COPYRIGHT: Copyright (C) 2021-2022, pyDecNef
# URL: https://pedromargolles.github.io/pyDecNef/
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################

"""
Description: Real-time fMRI data processing and decoding pipeline

This module serves as the main entry point for the real-time fMRI data processing 
pipeline, handling server communication, volume preprocessing, and trial management.

The pipeline processes fMRI volumes in real-time, enabling efficient data handling,
preprocessing, and decoding while maintaining asynchronous processing to avoid blocking.

Key features:
    - Server initialization and request handling
    - Volume acquisition and preprocessing
    - Trial management and decoding
    - Data logging and console reporting
    - Heatup and baseline correction handling

Functions:
    1. listen(): Starts listening for client requests in an independent thread
    2. process_volume(): Handles volume acquisition, labeling, and preprocessing
    3. manage_trials(): Manages trial creation, assignment, and decoding
    4. handle_preprocessing(): Coordinates volume preprocessing in separate threads
    5. console_report(): Provides real-time feedback on processing status
"""

import os
import time
from colorama import init, Fore
import threading
from modules.config import shared_instances
init(autoreset=True)

#############################################################################################
# SET EXPERIMENTAL PARAMETERS
#############################################################################################
total_probs = []

from modules.config.exp_config import Exp
Exp._new_participant()

#############################################################################################
# CLEAR SCREEN
#############################################################################################

clear = lambda: os.system('clear')
clear()

#############################################################################################
# INITIALIZE SERVER FOR CLIENT COMMUNICATION
#############################################################################################

from modules.config.connection_config import Connection
shared_instances.server = Connection()
shared_instances.server.start_server()

#############################################################################################
# INSTANTIATE TIMESERIES OBJECT
#############################################################################################

from modules.classes.classes import Timeseries
shared_instances.timeseries = Timeseries()

#############################################################################################
# INSTANTIATE FIRST TRIAL OBJECT
#############################################################################################

from modules.classes.classes import Trial
shared_instances.new_trial = Trial()

#############################################################################################
# INITIALIZE VOLUMES WATCHER FOR INCOMING DICOMS
#############################################################################################

from modules.classes.classes import Watcher
watcher = Watcher()
watcher.empty_fMRI_folder()  # Remove any leftover volumes from previous runs

#############################################################################################
# START RECEIVING FLOW CONTROL REQUESTS FROM THE EXPERIMENTAL SOFTWARE
#############################################################################################

from modules.config.listener import Listener
listener = Listener()
listener.listen()

#############################################################################################
# INITIALIZE VOLUMES LOGGER
#############################################################################################

from modules.classes.classes import Logger
shared_instances.logger = Logger()

#############################################################################################
# MAIN LOOP
#############################################################################################

print(Fore.YELLOW + '\n[START] Listening for new volumes...')

from modules.classes.classes import Vol
vol_idx = Exp.first_vol_idx

while True:

    new_vol = Vol(vol_idx=vol_idx)

    print('.....................................................................')

    new_vol.dicom_file = watcher.vol_watcher(new_vol)

    time.sleep(0.1)  # Brief pause to ensure DICOM file is fully written via Samba

    #############################################################################################
    # MRI SCANNER HEATUP
    #############################################################################################

    heatup_end = Exp.first_vol_idx + Exp.n_heatup_vols - 1

    if new_vol.vol_idx <= heatup_end:
        new_vol.vol_type = 'heatup'
        print(Fore.RED + '[HEATING UP] MRI scanner is heating up.')

    if new_vol.vol_idx == heatup_end:
        shared_instances.server.send('fmriheatedup')

    #############################################################################################
    # fMRI BASELINE
    #############################################################################################

    baseline_end = heatup_end + Exp.n_baseline_vols

    if heatup_end < new_vol.vol_idx <= baseline_end:
        new_vol.vol_type = 'baseline'
        print(Fore.RED + '[BASELINE] Measuring this fMRI run baseline activity.')

    if new_vol.vol_idx == baseline_end:
        shared_instances.server.send('baselineok')

    #############################################################################################
    # TASK STARTS
    #############################################################################################

    if new_vol.vol_idx > baseline_end:
        new_vol.vol_type = 'task'

    #############################################################################################
    # VOLUME PREPROCESSING
    #############################################################################################

    new_vol.preprocessing()
    shared_instances.new_trial.assign(new_vol)

    #############################################################################################
    # CONSOLE REPORT
    #############################################################################################

    if shared_instances.new_trial.trial_idx is not None:
        # Track unique decoding probabilities
        if shared_instances.new_trial.decoding_prob is not None:
            if not total_probs or shared_instances.new_trial.decoding_prob != total_probs[-1]:
                total_probs.append(shared_instances.new_trial.decoding_prob)

        print(Fore.YELLOW + '\nSubject:', Exp.subject, Fore.YELLOW + 'Session:', Exp.session, Fore.YELLOW + 'Run:', Exp.run)
        print(Fore.YELLOW + 'Trial:', shared_instances.new_trial.trial_idx)
        print(Fore.YELLOW + 'Trial onset time:', shared_instances.new_trial.trial_onset)
        print(Fore.YELLOW + 'Total probs:\n', total_probs)
        print(Fore.YELLOW + 'Volume index:', new_vol.vol_idx)
        print(Fore.YELLOW + 'Volume time:', new_vol.vol_time)
        print(Fore.YELLOW + 'Volume type:', new_vol.vol_type)
        print(Fore.YELLOW + 'Time after trial onset:', new_vol.vol_vs_trial_onset)
        print(Fore.YELLOW + f'Is this volume within HRF Peak ({Exp.HRF_peak_onset}-{Exp.HRF_peak_offset}s from trial onset):', new_vol.in_hrf_peak)
        print(Fore.YELLOW + f'Has HRF peak already ended for this trial?:', shared_instances.new_trial.HRF_peak_end)
        print(Fore.YELLOW + f'Number of volumes within HRF peak in this trial:', shared_instances.new_trial.n_HRF_peak_vols)
        print(Fore.YELLOW + f'Trial decoding prob:', shared_instances.new_trial.decoding_prob)
        print(Fore.YELLOW + f'Has decoding finished?:', shared_instances.new_trial.decoding_done)
        print(Fore.YELLOW + f'Number of current threads:', threading.active_count())
    else:
        print(Fore.YELLOW + 'Waiting for the first trial...')

    #############################################################################################
    # PREPARE FOR NEXT VOL
    #############################################################################################
    vol_idx += 1