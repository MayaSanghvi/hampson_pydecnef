############################################################################
# AUTHORS: Pedro Margolles & David Soto & Najemeddine Abdennour
# EMAIL: pmargolles@bcbl.eu, dsoto@bcbl.eu
# COPYRIGHT: Copyright (C) 2024-2025, pyDecNef
# URL: https://pedromargolles.github.io/pyDecNef/
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
from modules.config import shared_instances
from modules.config.exp_config import Exp
from modules.pipelines.corregistration_pipeline import corregister_vol
from modules.pipelines.preproc_vol_to_timeseries_pipeline import preproc_to_baseline, preproc_to_timeseries, preproc_to_model_session
from modules.pipelines.trial_decoding_pipeline import average_hrf_peak_vols_decoding, average_probs_decoding, dynamic_decoding
from colorama import Fore
import numpy as np
import pandas as pd
from pathlib import Path
import threading
import time
import pickle

# SET VARIABLES
coadaptation_time_hist = []


class Vol(Exp):
    """
    Class Object containing all information relative to each volume
    (timings, indexes, type, file path, preprocessing status...)
    and calls for corregistration and preprocessing functions.
    """
    def __init__(self, vol_idx=None):
        self.vol_idx = vol_idx
        self.trial = None
        self.vol_time = None
        self.vol_type = None
        self.dicom_file = None
        self.vol_vs_trial_onset = None
        self.in_hrf_peak = False
        self.data = None
        self.corregistration_times = None
        self.preproc_vol_to_timeseries_times = None
        self.prepared_4_decoding = False
        self.decoding_prob = None
        self.decoding_time = None

    def preprocessing(self):
        """Launch volume preprocessing in a separate thread to avoid blocking the main watcher loop."""
        preprocess_vol_thread = threading.Thread(
            name='vol_preprocessing',
            target=self._start_preprocessing,
        )
        preprocess_vol_thread.start()

    def _start_preprocessing(self):
        self.data, self.corregistration_times = corregister_vol(
            vol_file=self.dicom_file,
            vol_idx=self.vol_idx,
            mask_file=self.mask_file,
            ref_vol_file=self.ref_vol_file,
            preprocessed_dir=self.preprocessed_dir,
        )

        shared_instances.timeseries.preproc_vol_2_timeseries(self)

        self.prepared_4_decoding = True
        self._store_vol()

    def _store_vol(self):
        """Save a preprocessed version of the volume using vol_idx for a clean filename."""
        vol_name = f'vol_{self.vol_idx:04d}'
        np.save(
            Path(self.preprocessed_dir) / f'{vol_name}.npy',
            self.data,
            allow_pickle=True,
        )


class Timeseries(Exp):
    """
    Contains all masked volumes and functions for preprocessing
    (i.e., detrending, zscoring, high-pass filtering) each task volume
    in relation to previous volumes and baseline.
    """
    def __init__(self):
        self.heatup_vols = np.array([])
        self.baseline_vols = np.array([])
        self.task_vols = np.array([])
        self.whole_timeseries = np.array([])

    def preproc_vol_2_timeseries(self, vol):
        """Process volumes to timeseries."""
        self._append_vol(vol)

        if vol.vol_type == 'task':
            if self.zscoring_procedure == 'to_baseline':
                vol.data, vol.preproc_vol_to_timeseries_time = preproc_to_baseline(
                    whole_timeseries=self.whole_timeseries,
                    baseline_vols=self.baseline_vols,
                    preprocessed_dir=self.preprocessed_dir,
                )

            elif self.zscoring_procedure == 'to_timeseries':
                vol.data, vol.preproc_vol_to_timeseries_time = preproc_to_timeseries(
                    whole_timeseries=self.whole_timeseries,
                )

            elif self.zscoring_procedure == 'to_model_session':
                vol.data, vol.preproc_vol_to_timeseries_time = preproc_to_model_session(
                    whole_timeseries=self.whole_timeseries,
                    zscoring_mean=self.zscoring_mean_file,
                    zscoring_std=self.zscoring_std_file,
                )

    def _append_vol(self, vol):
        """Stack new masked volume onto baseline_vols, task_vols or whole_timeseries arrays."""

        def vol_to_array(timeseries_array, vol):
            """Append masked volume array to specific array."""
            if timeseries_array.shape[0] == 0:
                timeseries_array = vol
            else:
                timeseries_array = np.vstack([timeseries_array, vol])
            return timeseries_array

        if vol.vol_type == 'heatup':
            pass

        elif vol.vol_type == 'baseline':
            self.baseline_vols = vol_to_array(self.baseline_vols, vol.data)
            np.save(Path(self.preprocessed_dir) / 'unpreprocessed_baseline.npy', self.baseline_vols)

            self.whole_timeseries = vol_to_array(self.whole_timeseries, vol.data)
            np.save(Path(self.preprocessed_dir) / 'unpreprocessed_whole_timeseries.npy', self.whole_timeseries)

        elif vol.vol_type == 'task':
            self.task_vols = vol_to_array(self.task_vols, vol.data)
            np.save(Path(self.preprocessed_dir) / 'unpreprocessed_task_vols.npy', self.task_vols)

            self.whole_timeseries = vol_to_array(self.whole_timeseries, vol.data)
            np.save(Path(self.preprocessed_dir) / 'unpreprocessed_whole_timeseries.npy', self.whole_timeseries)

        else:
            print(Fore.RED + 'Volume type is not defined.')


class Trial(Exp):
    """
    Contains all information (timings, indexes, ground truth, stimuli, decoding probability...)
    relative to each experimental trial and fMRI volumes within them.
    """
    def __init__(self, trial_idx=None, trial_onset=None, stimuli=None, ground_truth=None):
        self.trial_idx = trial_idx
        self.trial_onset = trial_onset
        self.stimuli = stimuli
        self.ground_truth = ground_truth
        self.vols = []
        self.HRF_peak_vols = []
        self.n_HRF_peak_vols = 0
        self.HRF_peak_end = False
        self.decoding_prob = None
        self.decoding_time = None
        self.decoding_done = False

    def assign(self, vol):
        """Assign metadata information to volumes, selecting which volumes to decode."""
        if self.trial_idx is not None:
            vol.vol_vs_trial_onset = vol.vol_time - self.trial_onset
            self.vols.append(vol)
            vol.trial = self

            if self.HRF_peak_onset <= vol.vol_vs_trial_onset <= self.HRF_peak_offset:
                vol.in_hrf_peak = True
                self.HRF_peak_vols.append(vol)
                self.n_HRF_peak_vols = len(self.HRF_peak_vols)

            if (vol.vol_vs_trial_onset + self.TR) > self.HRF_peak_offset:
                self.HRF_peak_end = True

            self._store_trial()

        else:
            self.vols.append(vol)
            vol.trial = self
            self._store_trial()

        shared_instances.logger.add_vol(vol)

    def _decode(self):
        """Initiate the decoding process for the selected volumes of each trial."""

        if self.decoding_procedure == 'average_probs':
            while (self.HRF_peak_end == False) or (self.HRF_peak_vols[-1].prepared_4_decoding == False):
                time.sleep(0.1)

            preproc_vols_data = [vol.data for vol in self.HRF_peak_vols]

            trial_decoding_prob, vols_decoding_probs, trial_decoding_time = average_probs_decoding(
                preproc_vols_data=preproc_vols_data,
                model_file=self.model_file,
                ground_truth=self.ground_truth,
            )

            shared_instances.server.send(trial_decoding_prob)
            time.sleep(0.05)

            self.decoding_prob = trial_decoding_prob
            self.decoding_time = trial_decoding_time

            for vol, vol_decoding_prob in zip(self.HRF_peak_vols, vols_decoding_probs):
                vol.decoding_prob = vol_decoding_prob
                shared_instances.logger.update_vol(vol)

            self.decoding_done = True

        elif self.decoding_procedure == 'average_hrf_peak_vols':
            while (self.HRF_peak_end == False) or (self.HRF_peak_vols[-1].prepared_4_decoding == False):
                time.sleep(0.1)

            preproc_vols_data = [vol.data for vol in self.HRF_peak_vols]

            trial_decoding_prob, trial_decoding_time = average_hrf_peak_vols_decoding(
                preproc_vols_data=preproc_vols_data,
                model_file=self.model_file,
                ground_truth=self.ground_truth,
            )

            shared_instances.server.send(trial_decoding_prob)
            time.sleep(0.05)

            self.decoding_prob = trial_decoding_prob
            self.decoding_time = trial_decoding_time
            self.decoding_done = True

        elif self.decoding_procedure == 'dynamic':
            last_decoded_vol = None

            while self.HRF_peak_end == False:
                if (len(self.HRF_peak_vols) >= 1) and (self.HRF_peak_vols[-1].prepared_4_decoding == True):
                    if id(last_decoded_vol) != id(self.HRF_peak_vols[-1]):
                        last_vol = self.HRF_peak_vols[-1]

                        vol_decoding_prob, vol_decoding_time = dynamic_decoding(
                            last_vol.data,
                            self.model_file,
                            self.ground_truth,
                        )

                        shared_instances.server.send(vol_decoding_prob)
                        time.sleep(0.05)

                        last_vol.decoding_prob = vol_decoding_prob
                        last_vol.decoding_time = vol_decoding_time
                        shared_instances.logger.update_vol(last_vol)
                        last_decoded_vol = last_vol
                    else:
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)

            self.decoding_done = True

        # Coadaptation (runs for all decoding procedures)
        if Exp.coadaptation_background_warmup or Exp.coadaptation_active:
            self.saved_preproc_vols_data = [vol.data for vol in self.HRF_peak_vols]
            self.saved_ground_truth = self.ground_truth
            coadaptation_thread = threading.Thread(
                name="coadaptation",
                target=self._decoder_coadaptation,
            )
            coadaptation_thread.start()

    def _decoder_coadaptation(self):
        """Initiate decoding co-adaptation."""
        from modules.pipelines import decoding_coadaptation
        start_coadaptation_timer = time.perf_counter()
        decoding_coadaptation.coadaptation(
            self.saved_preproc_vols_data,
            self.saved_ground_truth,
            Exp.model_file,
        )
        end_coadaptation_timer = time.perf_counter()
        coadaptation_timer = end_coadaptation_timer - start_coadaptation_timer
        coadaptation_time_hist.append(coadaptation_timer)
        print(
            f"The coadaptation process took: {coadaptation_timer:.3f} s\n"
            f"Mean coadaptation time: {np.mean(coadaptation_time_hist):.3f} s, "
            f"Max coadaptation time: {np.max(coadaptation_time_hist):.3f} s"
        )

    def _store_trial(self):
        """Store the trial with all preprocessed volumes as a pickled object."""
        try:
            trial_file = open(Path(self.trials_dir) / f'trial_{self.trial_idx}.pkl', 'wb')
            pickle.dump(self, trial_file)
            trial_file.close()
        except:
            print("trial pickle could not be saved")


class Watcher(Exp):
    """
    Watches for newly arriving DICOM files from the scanner.

    Instead of expecting sequentially-named files (e.g., IM-0001.dcm),
    this watcher monitors the raw volumes folder for any new files and
    orders them by arrival time. Compatible with Siemens DICOM naming
    conventions (e.g., MRe.1.3.12.2...).
    """
    def __init__(self):
        self._seen_files = set()

    def empty_fMRI_folder(self):
        """Empty fMRI raw volumes output folder."""
        folder_exist = self._check_folder()
        if folder_exist:
            print(Fore.CYAN + '\n[WAIT] Removing all volumes from MRI folder...')
            raw_volumes_folder = Path(self.raw_volumes_folder)
            for file in raw_volumes_folder.iterdir():
                if file.is_file():
                    file.unlink()
            print(Fore.GREEN + f'[OK] Volumes removed.')

    def vol_watcher(self, new_vol):
        """
        Watch for a newly generated volume file.

        Detects new files by comparing the current directory contents
        against previously seen files, rather than relying on a specific
        filename pattern. Returns the path to the new DICOM file.
        """
        print(Fore.CYAN + f'\n[WAIT] Waiting for volume {new_vol.vol_idx}...')

        while True:
            current_files = set(Path(self.raw_volumes_folder).iterdir())
            new_files = current_files - self._seen_files

            if new_files:
                new_file = min(new_files, key=lambda f: f.stat().st_mtime)
                self._seen_files.add(new_file)
                new_vol.vol_time = time.time()
                dicom_file = str(new_file)
                print(Fore.GREEN + f'[OK] Vol {new_vol.vol_idx} received: {new_file.name}')
                return dicom_file

            time.sleep(0.05)

    def _check_folder(self):
        """Check that the MRI folder exists."""
        print(Fore.CYAN + '\n[WAIT] Checking MRI folder to watch...')
        if Path(self.raw_volumes_folder).is_dir():
            print(Fore.GREEN + f'[OK] Folder OK.')
            return True
        else:
            print(Fore.RED + f'[ERROR] MRI folder "{self.raw_volumes_folder}" does not exist.')
            return False


class Logger(Exp):
    """
    Logs data relative to timings, trials, volumes and decoding results into a CSV file.
    """
    def __init__(self):
        self.this_run_vols = []
        self.list_rows = []

    def add_vol(self, vol):
        """Log a new volume's information."""
        self.this_run_vols.append(vol)
        variables_to_log = self._log_variables(vol)
        self.list_rows.append(variables_to_log)
        df = pd.DataFrame(self.list_rows)
        df.to_csv(Path(self.logs_dir) / f'logs.csv')

    def update_vol(self, vol):
        """Update the information of an already logged volume."""
        vol_idx = self.this_run_vols.index(vol)
        variables_to_log = self._log_variables(vol)
        self.list_rows[vol_idx] = variables_to_log
        df = pd.DataFrame(self.list_rows)
        df.to_csv(Path(self.logs_dir) / f'logs.csv')

    def _log_variables(self, vol):
        """Specify which variables to log."""
        variables_to_log = {
            'subject': Exp.subject,
            'session': Exp.session,
            'run': Exp.run,
            'n_heatup_vols': Exp.n_heatup_vols,
            'HRF_peak_onset': Exp.HRF_peak_onset,
            'HRF_peak_offset': Exp.HRF_peak_offset,
            'vol_type': vol.vol_type,
            'trial_idx': vol.trial.trial_idx,
            'trial_onset': vol.trial.trial_onset,
            'vol_idx': vol.vol_idx,
            'vol_time': vol.vol_time,
            'vol_time_vs_trial_onset': vol.vol_vs_trial_onset,
            'in_hrf_peak': vol.in_hrf_peak,
            'decoding_procedure': Exp.decoding_procedure,
            'vol_decoding_prob': vol.decoding_prob,
            'trial_decoding_prob': vol.trial.decoding_prob,
            'stimuli': vol.trial.stimuli,
            'ground_truth': vol.trial.ground_truth,
        }
        return variables_to_log