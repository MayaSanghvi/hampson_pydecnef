############################################################################
# AUTHORS: Pedro Margolles & David Soto & Najemeddine Abdennour
# EMAIL: pmargolles@bcbl.eu, dsoto@bcbl.eu
# COPYRIGHT: Copyright (C) 2024-2025, pyDecNef
# URL: https://pedromargolles.github.io/pyDecNef/
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
"""
Sets the configuration and parameters of the realtime fMRI procedure 
read from the config.ini file.
"""
from pathlib import Path
import time
from colorama import init, Fore
import sys
import os
import configparser
from joblib import load, dump
init(autoreset=True)

#############################################################################################
# LOAD CONFIG FILE
#############################################################################################

config_file = os.path.join(Path(__file__).absolute().parent.parent.parent.parent, "config.ini")
config = configparser.ConfigParser()
config.read(config_file)


class Exp:
    #############################################################################################
    # EXPERIMENTAL PARAMETERS
    #############################################################################################

    # Volumes processing information
    n_heatup_vols = int(config["experiment"]["n_heatup_vols"])
    n_baseline_vols = int(config["experiment"]["n_baseline_vols"])
    HRF_peak_onset = int(config["experiment"]["HRF_peak_onset"])
    HRF_peak_offset = int(config["experiment"]["HRF_peak_offset"])
    TR = int(config["experiment"]["TR"])

    # Volumes tracking
    first_vol_idx = int(config["experiment"]["first_vol_idx"])
    index_format = config["experiment"]["index_format"]  # NOTE: retained for compatibility but no longer used by the Watcher for file detection

    # Z-scoring procedure
    zscoring_procedure = config["experiment"]["zscoring_procedure"]

    # Decoding settings
    decoding_procedure = config["experiment"]["decoding_procedure"]
    coadaptation_active = config.getboolean("experiment", "coadaptation_active")
    coadaptation_background_warmup = config.getboolean("experiment", "coadaptation_background_warmup")
    coadaptation_vol_acceptance_criteria = float(config["experiment"]["coadaptation_vol_acceptance_criteria"])
    classifier_type = config["experiment"]["classifier_type"]

    @classmethod
    def _new_participant(cls):
        """Request new participant data (participant, session, run) each time main.py runs to set directory routes."""

        def check_file(file):
            """Check if an essential file exists. If not, cancel script execution."""
            if not os.path.exists(file):
                sys.exit(Fore.RED + f'[ERROR] File/Directory "{file}" does not exist. Check that you are pointing to a correct path.')

        #############################################################################################
        # DIRECTORIES & DATA
        #############################################################################################

        print(Fore.YELLOW + 'Specify participant data before initialization:')
        cls.subject = input(Fore.YELLOW + '\nParticipant number: ')
        cls.session = input(Fore.YELLOW + 'Session number: ')
        cls.run = input(Fore.YELLOW + 'Run number: ')
        print('\n')

        # Package directory
        cls.moduledir = Path(__file__).absolute().parent.parent.parent

        # fMRI raw volumes output folder
        if config.getboolean("experiment", "simulated_experiment"):
            cls.raw_volumes_folder_path = os.path.join(cls.moduledir.parent, '2_MRI_simulator', 'outputs')
        else:
            cls.raw_volumes_folder_path = config["files_and_dir"]["raw_volumes_folder_path"]
        cls.raw_volumes_folder = Path(cls.raw_volumes_folder_path)
        check_file(cls.raw_volumes_folder)

        # Required resources directory
        cls.resources_dir = os.path.join(cls.moduledir.parent, 'required_resources', f'sub-{cls.subject}')
        check_file(cls.resources_dir)

        # Pretrained model path
        cls.model_name = config["files_and_dir"]["model_name"]
        cls.model_file = os.path.join(cls.resources_dir, 'models', cls.model_name)
        check_file(cls.model_file)

        if cls.coadaptation_active:
            cls.coadaptation_model_name = cls.model_name + "_coadapted"
            cls.coadaptation_model_file = os.path.join(cls.resources_dir, 'models', cls.coadaptation_model_name)
            if not os.path.exists(cls.coadaptation_model_file):
                model = load(cls.model_file)
                dump(model, cls.coadaptation_model_file)
            cls.model_file = cls.coadaptation_model_file

        # Region of interest mask path
        cls.mask_name = config["files_and_dir"]["mask_name"]
        cls.mask_file = os.path.join(cls.resources_dir, 'masks', cls.mask_name)
        check_file(cls.mask_file)

        # Reference functional volume path
        cls.ref_vol_name = config["files_and_dir"]["ref_vol_name"]
        cls.ref_vol_file = os.path.join(cls.resources_dir, 'training_session_ref_image', cls.ref_vol_name)
        check_file(cls.ref_vol_file)

        # ROI reference data for z-scoring
        if cls.zscoring_procedure == 'to_model_session':
            cls.zscoring_mean_file_name = config["files_and_dir"]["zscoring_mean_file_name"]
            cls.zscoring_std_file_name = config["files_and_dir"]["zscoring_std_file_name"]
            cls.zscoring_mean_file = os.path.join(cls.resources_dir, 'training_zscoring_data', cls.zscoring_mean_file_name)
            cls.zscoring_std_file = os.path.join(cls.resources_dir, 'training_zscoring_data', cls.zscoring_std_file_name)

        # Coadaptation training data
        if cls.coadaptation_active or cls.coadaptation_background_warmup:
            cls.coadaptation_base_training_data_dir_name = config["files_and_dir"]["coadaptation_base_training_data_dir_name"]
            cls.coadaptation_training_data_file_name = config["files_and_dir"]["coadaptation_training_data_file_name"]
            cls.coadaptation_training_data_labels_file_name = config["files_and_dir"]["coadaptation_training_data_labels_file_name"]
            cls.coadaptation_training_data_dir = os.path.join(cls.resources_dir, cls.coadaptation_base_training_data_dir_name)
            cls.coadaptation_training_data_file = os.path.join(cls.coadaptation_training_data_dir, cls.coadaptation_training_data_file_name)
            cls.coadaptation_training_data_labels_file = os.path.join(cls.coadaptation_training_data_dir, cls.coadaptation_training_data_labels_file_name)
            check_file(cls.coadaptation_training_data_dir)
            check_file(cls.coadaptation_training_data_file)
            check_file(cls.coadaptation_training_data_labels_file)

        # Output directories
        cls.outputs_dir = os.path.join(cls.moduledir, 'outputs', f'sub-{cls.subject}_session-{cls.session}')
        Path(cls.outputs_dir).mkdir(parents=True, exist_ok=True)

        # main.py script run time
        script_run_time = time.strftime('%Y-%m-%d_%H-%M-%S')

        # Make a run directory inside outputs dir to store all participant log files and preprocessed volumes
        cls.run_dir = os.path.join(cls.outputs_dir, f'run-{cls.run}_{script_run_time}')
        Path(cls.run_dir).mkdir(parents=True, exist_ok=True)

        # Make a trials directory inside run directory to store all masked volumes and information classified by trial
        cls.trials_dir = os.path.join(cls.run_dir, 'trials')
        Path(cls.trials_dir).mkdir(parents=True, exist_ok=True)

        # Make a logs directory inside run directory to store run logs data
        cls.logs_dir = os.path.join(cls.run_dir, 'logs_dir')
        Path(cls.logs_dir).mkdir(parents=True, exist_ok=True)

        # Make a preprocessed volumes directory inside run directory to store all outputs corresponding to preprocessed volumes in that run
        cls.preprocessed_dir = os.path.join(cls.run_dir, 'preprocessed')
        Path(cls.preprocessed_dir).mkdir(parents=True, exist_ok=True)