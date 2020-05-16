from MNE_Pipeline import MNE_Repo_Mat
import numpy as np
from multiprocessing import Process, Manager
import os
import mne
import sys
from sklearn.cluster import KMeans

class MNE_Repo_Mat_ext(MNE_Repo_Mat):

    def __init__(self):
        super()

    def combine_events_and_save(self, epochs, ids, new_ids, subject):
        # events = epochs.events
        # new_events = mne.merge_events(events, ids, new_ids)
        # tmin = epochs.tmin
        #
        # new_epochs = mne.EpochsArray()
        folder_name = '1_5_trigg_combined_epochs'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        save_path = '1_5_trigg_combined_epochs/{}.fif'.format(subject)
        new_epochs = mne.epochs.combine_event_ids(epochs, ids, new_ids, True)
        new_epochs.save(save_path)

    def bootstrap_epochs_with_RT(self, event_id, epoch, sampling_rate, iterations, dict_holder, possible_combinations = None):
        bst_sample = []
        if possible_combinations == None:
            self.bootstrap_epochs(event_id, epoch, sampling_rate, iterations, dict_holder)
            return
        elif len(possible_combinations) == 0:
            r_erp = np.average(epoch, axis=0)
            bst_sample.append(r_erp)
            bst_sample = np.array(bst_sample)
        else:
            for i,c in enumerate(possible_combinations):
                c = list(c)
                r_epoch = epoch[c]
                r_erp = np.average(r_epoch, axis=0)
                if i == 0:
                    bst_sample.append(r_erp)
                    bst_sample = np.array(bst_sample)
                    continue
                bst_sample = np.append(bst_sample, r_erp.reshape((1, 64, 500)), axis=0)
        dict_holder[event_id] = bst_sample

    def bootstrap_epochs(self, event_id, epoch, sampling_rate, iterations, dict_holder):
        bst_sample = []
        for i in range(iterations):
            # print(i)
            # sys.stdout.flush()
            r_sample = np.random.choice(list(range(len(epoch))), size=sampling_rate)
            r_epoch = epoch[r_sample]
            r_erp = np.average(r_epoch, axis=0)
            if i == 0:
                bst_sample.append(r_erp)
                bst_sample = np.array(bst_sample)
                continue
            bst_sample = np.append(bst_sample, r_erp.reshape((1, 64, 500)), axis=0)
        dict_holder[event_id] = bst_sample

    def async_bootstrap_epochs_with_RT(self, subject, epochs, event_ids, sampling_rate=10, iterations=300, save_path='bootstrap_erps_cl_vs_amb'):
        import pickle
        import math
        from itertools import combinations
        def nCr(n,r):
            if r > n:
                return 0
            f = math.factorial
            return f(n) // (f(r) * f(n-r))
        save_path = save_path + '/'
        subject_save_path = save_path + subject

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        manager = Manager()

        subject_bt_erp = dict()

        processes = []
        for event in epochs.event_id:
            if event not in event_ids:
                continue
            epoch = epochs[event].get_data()
            no_of_combinations = nCr(len(epoch), sampling_rate)
            if no_of_combinations < iterations:
                possible_combinations = list(combinations(range(len(epoch)), sampling_rate))
            else:
                possible_combinations = None
            self.bootstrap_epochs_with_RT(event, epoch, sampling_rate, iterations, subject_bt_erp, possible_combinations)
            # process = Process(target=self.bootstrap_epochs, args=(event, epoch, sampling_rate, iterations, subject_bt_erp))
            # processes.append(process)
            # process.start()

        # for process in processes:
        #     process.join()

        with open(subject_save_path, 'wb') as file:
            pickle.dump(subject_bt_erp, file)



    def async_bootstrap_epochs(self, subject, epochs, event_ids, sampling_rate=10, iterations=300, save_path='bootstrap_erps_cl_vs_amb'):
        import pickle
        save_path = save_path + '/'
        subject_save_path = save_path + subject

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        manager = Manager()

        subject_bt_erp = dict()

        processes = []
        for event in epochs.event_id:
            if event not in event_ids:
                continue
            epoch = epochs[event].get_data()
            self.bootstrap_epochs(event, epoch, sampling_rate, iterations, subject_bt_erp)
            # process = Process(target=self.bootstrap_epochs, args=(event, epoch, sampling_rate, iterations, subject_bt_erp))
            # processes.append(process)
            # process.start()
            print('Starting for {}_{}'.format(subject, event))

        # for process in processes:
        #     process.join()

        with open(subject_save_path, 'wb') as file:
            pickle.dump(subject_bt_erp, file)
