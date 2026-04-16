############################################################################
# AUTHORS: Pedro Margolles & David Soto
# EMAIL: pmargolles@bcbl.eu, dsoto@bcbl.eu
# COPYRIGHT: Copyright (C) 2021-2022, pyDecNef
# URL: https://pedromargolles.github.io/pyDecNef/
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
"""
This module implements a listener class capable of handling client requests in an 
asynchronous manner, ensuring that multiple operations can be processed without blocking
each other. The listener class is designed to support flexible request-action pairings,
allowing for customizable experimental setups.
"""
import os
import time
import threading
from colorama import Fore
from modules.classes.classes import Trial
from modules.config import shared_instances

#############################################################################################
# LISTENER CLASS
#############################################################################################

class Listener:
    def __init__(self):
        pass

    def listen(self):
        listener_thread = threading.Thread(name='listener',
                                           target=self._start_listen)
        listener_thread.start()

    def _start_listen(self):
        while True:
            client_request = shared_instances.server.listen()

            process_requests_thread = threading.Thread(name='process_requests',
                                                       target=self._process_client_requests,
                                                       args=(client_request,))
            process_requests_thread.start()

    #############################################################################################
    # CLIENT REQUESTS - SERVER ACTIONS PAIRINGS
    #############################################################################################

    def _process_client_requests(self, client_request):
        if client_request['request_type'] == 'trial_onset':
            shared_instances.new_trial = Trial()
            shared_instances.new_trial.trial_idx = client_request['trial_idx']
            shared_instances.new_trial.ground_truth = client_request['ground_truth']
            shared_instances.new_trial.stimuli = client_request['stimuli']
            shared_instances.new_trial.trial_onset = time.time()
            shared_instances.server.send('ok')

        elif client_request['request_type'] == 'feedback_start':
            feedback_thread = threading.Thread(name='decoding_trial',
                                               target=shared_instances.new_trial._decode)
            feedback_thread.start()
            feedback_thread.join()

            if shared_instances.new_trial.decoding_done == True:
                shared_instances.server.send('ok')

        elif client_request['request_type'] == 'feedback_end':
            shared_instances.new_trial.HRF_peak_end = True

        elif client_request['request_type'] == 'end_run':
            print(Fore.GREEN + '[FINISHING] Experimental run is over.')
            shared_instances.server.send('ok')
            os._exit(1)

        else:
            print(Fore.RED + f'[ERROR] Request {client_request} not recognized.')