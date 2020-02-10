from MNE_Pipeline import MNE_Repo_Mat
import numpy as np
from multiprocessing import Process, Manager
import os

class MNE_Repo_Mat_ext(MNE_Repo_Mat):

    def __init__(self):
        super()

    def bootstrap_epochs(self, event_id, epoch, sampling_rate, iterations, dict_holder):
        bst_sample = []
        bst_RTs = []
        for i in range(iterations):
            r_sample = np.random.choice(list(range(len(epoch))),size=sampling_rate)
            r_epoch = epoch[r_sample]
            # r_RT = RT[r_sample]
            r_erp = np.average(r_epoch, axis=0)
            # r_avg_RT = np.average(r_RT)
            # # discrete_RT = kmn.predict([[r_avg_RT]])[0]
            # # if discrete_RT == 0:
            # #     discrete_RT = 2
            # # elif discrete_RT == 2:
            # #     discrete_RT = 0
            # bst_RTs.append((r_avg_RT, discrete_RT))
            if i == 0:
                bst_sample.append(r_erp)
                bst_sample = np.array(bst_sample)
                continue
            bst_sample = np.append(bst_sample, r_erp.reshape((1, 64, 500)), axis=0)

        dict_holder[event_id] = bst_sample

        with open('status.txt', 'a') as f:
            f.write('did it')
            f.write('/n')





    def async_bootstrap_epochs(self, subject, epochs, sampling_rate=10, iterations=300):
        import pickle
        save_path = 'bootstrap_erps/'
        subject_save_path = save_path + subject

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        manager = Manager()

        subject_bt_erp = manager.dict()

        processes = []
        for event in epochs.event_id:
            if event == '-1':
                continue
            epoch = epochs[event].get_data()
            # RT = RT_dict[event]
            process = Process(target=self.bootstrap_epochs, args=(event, epoch, sampling_rate, iterations, subject_bt_erp))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        with open(subject_save_path, 'wb') as file:
            pickle.dump(dict(subject_bt_erp), file)
