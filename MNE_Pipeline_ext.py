from MNE_Pipeline import MNE_Repo_Mat
import numpy as np
from multiprocessing import Process, Manager
import os
from sklearn.cluster import KMeans

class MNE_Repo_Mat_ext(MNE_Repo_Mat):

    def __init__(self):
        super()

    def bootstrap_epochs_with_RT(self, event_id, epoch, RT, sampling_rate, iterations, dict_holder, gmm):
        bst_sample = []
        bst_RTs = []
        for i in range(iterations):
            r_sample = np.random.choice(list(range(len(epoch))),size=sampling_rate)
            r_epoch = epoch[r_sample]
            r_RT = RT[r_sample]
            r_erp = np.average(r_epoch, axis=0)
            r_avg_RT = np.average(r_RT)
            discrete_RT = gmm.predict([[r_avg_RT]])[0]
            bst_RTs.append((r_avg_RT, discrete_RT))
            if i == 0:
                bst_sample.append(r_erp)
                bst_sample = np.array(bst_sample)
                continue
            bst_sample = np.append(bst_sample, r_erp.reshape((1, 64, 500)), axis=0)

        dict_holder[event_id] = (bst_sample, bst_RTs)
        
    def bootstrap_epochs(self, event_id, epoch, sampling_rate, iterations, dict_holder):
        bst_sample = []
        for i in range(iterations):
            r_sample = np.random.choice(list(range(len(epoch))),size=sampling_rate)
            r_epoch = epoch[r_sample]
            r_erp = np.average(r_epoch, axis=0)
            if i == 0:
                bst_sample.append(r_erp)
                bst_sample = np.array(bst_sample)
                continue
            bst_sample = np.append(bst_sample, r_erp.reshape((1, 64, 500)), axis=0)

        dict_holder[event_id] = bst_sample

    def async_bootstrap_epochs_with_RT(self, subject, epochs, RT_dict, sampling_rate=10, iterations=300):
        ## Need re-implementation
        
        import pickle
        save_path = 'bootstrap_erps/'
        subject_save_path = save_path + subject

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        manager = Manager()

        subject_bt_erp = manager.dict()
        
        model_file = open('Bayesian_GMM_RT_model.pkl', 'rb')
        gmm = pickle.load(model_file)
        model_file.close()

        processes = []
        for event in epochs.event_id:
            if event != '15' and event != '3':
                continue
            epoch = epochs[event].get_data()
            RT = RT_dict[event]
            process = Process(target=self.bootstrap_epochs_with_RT, args=(event, epoch, RT, sampling_rate, iterations, subject_bt_erp, gmm))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        with open(subject_save_path, 'wb') as file:
            pickle.dump(dict(subject_bt_erp), file)
            
            
            
    def async_bootstrap_epochs(self, subject, epochs, sampling_rate=10, iterations=300):
        import pickle
        save_path = 'bootstrap_erps_cl_vs_amb/'
        subject_save_path = save_path + subject

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        manager = Manager()

        subject_bt_erp = manager.dict()

        processes = []
        for event in epochs.event_id:
            if event != '15' and event != '3':
                continue
            epoch = epochs[event].get_data()
            process = Process(target=self.bootstrap_epochs, args=(event, epoch, sampling_rate, iterations, subject_bt_erp))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        with open(subject_save_path, 'wb') as file:
            pickle.dump(dict(subject_bt_erp), file)
