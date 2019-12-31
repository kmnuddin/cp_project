import mne
import numpy as np
from scipy.io import loadmat
import os
from mne.datasets import fetch_fsaverage
import re

class MNE_Repo_Mat:
    subjects_dir = '';
    subject = '';
    bem_sol = mne.bem.ConductorModel()

    def load_data(self, filename):
        self.data_mat = loadmat(filename, squeeze_me=True, struct_as_record=False)
        self.behavResp = self.data_mat['behavResp']
        self.RT = self.data_mat['RT']
        self.trigs = self.data_mat['trigs']
        self.epochs_raw = self.data_mat['epochs'].transpose(2,0,1)
        self.t = self.data_mat['t']
        self.Fs = self.data_mat['Fs']
        self.NumChannels = self.data_mat['NumChannels']
        self.chanNames = self.data_mat['chanNames'].tolist()
        return self.data_mat

    @staticmethod
    def construct_subject():
        MNE_Repo_Mat.subjects_dir = os.path.dirname(fetch_fsaverage())
        MNE_Repo_Mat.subject='fsaverage'

    @staticmethod
    def construct_montage(kind, path):
        montage = mne.channels.read_montage(kind=kind, path=path, unit='auto', transform=False)
        montage.kind = '3d'
        montage.plot()
        return montage

    def construct_info(self, montage = None, sfreq = 500):
        if montage is None:
            montage = MNE_Repo_Mat.construct_montage('neuroscan64ch', 'montages')
        self.info = mne.create_info(montage.ch_names, sfreq, ch_types='eeg', montage=montage)
        return self.info


    def construct_events(self, trigs):
        number_of_trials = len(trigs)
        events = np.zeros((number_of_trials, 3), dtype=int)

        for i in range(len(trigs)):
            events[i,0] = i
            events[i,2] = trigs[i]
        return events


    def construct_epoch_array(self, tmin, events = None):
        self.epochs = mne.EpochsArray(self.epochs_raw, info=self.info, tmin=tmin, events=events)
        # self.event_ids = epochs.event_id
        return self.epochs

    def save_epochs(self, epochs):
        for key in epochs:
            epoch_path_to_save = 'epochs/' + key  + '.fif'
            epochs[key].save(epoch_path_to_save, overwrite=True)

    def load_epochs(self, path):
        self.epochs = mne.read_epochs(path)
        return self.epochs

    # def get_trigger_wise_epochs(self, epochs, event, event_ids):
    #     trig_wise_epochs = dict()
    #     trig_wise_epochs.keys = event_ids
    #     for epoch in epochs:

    def construct_evoked_array(self, method):
        evoked = self.epochs.average(method=method)
        evoked.set_eeg_reference(projection=True)
        evoked.apply_proj()
        evoked.plot(spatial_colors=True,unit=False)
        evoked.plot_topomap(times=[0.1], size=3)
        return evoked

    def construct_trigger_wise_evoked_array(self, epoch, event_ids, method):
        trig_wise_evoked = dict()
        for key in event_ids:
            evoked = epoch[key].average(method=method)
            # evoked.apply_baseline(baseline=(-0.2, 0))
            evoked.set_eeg_reference(projection=True)
            evoked.apply_proj()
            trig_wise_evoked[key] = evoked
            del evoked
        return trig_wise_evoked

    def save_trigger_wise_evokeds(self, evokeds):
        for key in evokeds:
            sub_folder_path = 'ERPs/' + key
            if not os.path.exists(sub_folder_path):
                os.mkdir(sub_folder_path)
            for event_id in evokeds[key]:
                erp_path_save = sub_folder_path + '/' + event_id + '_ave.fif'
                if os.path.exists(erp_path_save):
                    os.remove(erp_path_save)
                evokeds[key][event_id].save(erp_path_save)

    def load_trigger_wise_evokeds(self, folder_path, event_ids):
        trig_wise_evoked = dict()
        for key in event_ids:
            erp_path = folder_path + '/' + key + '_ave.fif'
            trig_wise_evoked[key] = mne.Evoked(erp_path)
        return trig_wise_evoked


    @staticmethod
    def setup_src_space():
        if not os.path.exists('source_space/src_space.fif'):
            src = mne.setup_source_space(MNE_Repo_Mat.subject, spacing='oct6')
            src.save('source_space/src_space.fif')
        else:
            src = mne.read_source_spaces('source_space/src_space.fif')
        return src

    @staticmethod
    def setup_bem():
        if not os.path.exists('bem/fsaverage_bem.fif'):
            model = mne.make_bem_model(MNE_Repo_Mat.subject)
            bem_sol = mne.make_bem_solution(model)
            mne.write_bem_solution('bem/fsaverage_bem.fif',bem_sol)
        else:
            bem_sol = mne.read_bem_solution('bem/fsaverage_bem.fif')
        return bem_sol

    @staticmethod
    def get_trans_obj():
        data_path = mne.datasets.sample.data_path()
        trans = mne.read_trans(data_path + '/MEG/sample/sample_audvis_raw-trans.fif')
        return trans

    def compute_forward_sol(self, info, src, bem):
        trans = MNE_Repo_Mat.get_trans_obj()
        fwd = mne.make_forward_solution(info=info, trans=trans, src=src, bem=bem)
        return fwd

    def compute_covariance_mat(self, epochs):
        return mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None)

    def create_inverse_operator(self, info, cov, fwd, loose, depth):
        return mne.minimum_norm.make_inverse_operator(info=info, noise_cov=cov, forward=fwd, loose=loose, depth=depth)

    def apply_inverse_operator_with_residual(self, evoked, inv, lambda2, ori, method, residual, verbose):
        stc, residual = mne.minimum_norm.apply_inverse(evoked, inv, lambda2,
                              method=method, pick_ori=ori,
                              return_residual=residual, verbose=verbose)
        return stc, residual

    def apply_inverse_operator(self, evoked, inv, lambda2, ori, method, verbose):
        stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2,
                              method=method, pick_ori=ori, verbose=verbose)
        return stc

    def apply_inverse_operator_event_wise(self, epoch, evoked, info, fwd, lambda2, ori, method, verbose):
        stc_single_sub = dict()

        for event_id in evoked:
            cov = self.compute_covariance_mat(epoch[event_id])
            inv = self.create_inverse_operator(info, cov, fwd, 0.2, 0.8)


            stc_single_sub[event_id] = self.apply_inverse_operator(evoked[event_id], inv, lambda2, ori, method, verbose)
        return stc_single_sub


    # def apply_inverse_operator_event_wise(self, evoked, inv, lambda2, ori, method, verbose):
    #     stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2,
    #                           method=method, pick_ori=ori, verbose=verbose)
    #     return stc

    @staticmethod
    def init_exp_for_sl():
        MNE_Repo_Mat.construct_subject()
        montage = MNE_Repo_Mat.construct_montage('neuroscan64ch', 'montages')
        src = MNE_Repo_Mat.setup_src_space()
        bem = MNE_Repo_Mat.setup_bem()
        return montage, src, bem

    def generate_source_estimate_straight(self, file_name, montage, src, bem):
        self.load_data(file_name)

        info = self.construct_info(montage)

        epochs = self.construct_epoch_array(-0.2)
        evoked = self.construct_evoked_array('mean')

        fwd = self.compute_forward_sol(info, src, bem)
        cov = self.compute_covariance_mat(epochs)
        inv = self.create_inverse_operator(info, cov, fwd, 0.2, 0.8)

        snr = 3.
        lambda2 = 1. / snr ** 2
        stc,residual = self.apply_inverse_operator_with_residual(evoked, inv, lambda2,  None, 'sLORETA', True, True)

        return stc,residual

    def save_event_wise_source_estimates(self, stcs):
        for stc_sub in stcs:
            stc_sub_path = 'stcs/' + stc_sub
            os.mkdir(stc_sub_path)
            for event_id in stcs[stc_sub]:
                event_stc_path = stc_sub_path + '/' + event_id
                stcs[stc_sub][event_id].save(fname = event_stc_path, ftype = 'stc')

    def load_stcs(self, path, subject):
        stc_sub_path = path + '/' + subject + '/'
        event_stcs = dict()
        for event_stc_file in os.listdir(stc_sub_path):
            if event_stc_file.endswith('.stc') and 'lh' in event_stc_file:
                event_key = re.findall(r'\d+', event_stc_file)[0]
                event_stc_path = stc_sub_path + event_stc_file
                event_stcs[event_key] = mne.read_source_estimate(event_stc_path)
        return event_stcs


    def generate_ERPs(self, filenames, montage, gen_mode = True, save = True):
        self.evokeds = dict()
        self.epochs_dict = dict()

        for filename in filenames:
            self.load_data(filename)
            info = self.construct_info(montage)
            erp_subject_key = re.split(r'[./]', filename)[1] + '_erp'
            epoch_subject_key = re.split(r'[./]', filename)[1] + '_epoch'

            events = self.construct_events(self.trigs)

            if gen_mode:
                self.epochs_dict[epoch_subject_key]  = self.construct_epoch_array(-0.2, events)
                self.evokeds[erp_subject_key] = self.construct_trigger_wise_evoked_array(self.epochs_dict[epoch_subject_key], self.epochs_dict[epoch_subject_key].event_id, 'mean')
            else:
                epoch_path = 'epochs/' + epoch_subject_key + '.fif'
                erp_folder_path = 'ERPs/' + erp_subject_key
                self.epochs_dict[epoch_subject_key] = self.load_epochs(epoch_path)
                self.evokeds[erp_subject_key] = self.load_trigger_wise_evokeds(erp_folder_path, self.epochs_dict[epoch_subject_key].event_id)

        if save:
            self.save_trigger_wise_evokeds(self.evokeds)
            self.save_epochs(self.epochs_dict)

        return self.evokeds, self.epochs_dict

    def generate_event_wise_stcs(self, epochs, evokeds, montage, src, bem, gen_mode = True, save = True):

        info = self.construct_info(montage)

        self.stcs = dict()

        for sub_key_epoch, sub_key_erp in zip(epochs, evokeds):

            sub_stc_key = re.split(r'[_]', sub_key_erp)[0] + '_stc'

            if not gen_mode:
                self.stcs[sub_stc_key] = self.load_stcs('stcs' ,sub_stc_key)
                continue

            fwd = self.compute_forward_sol(info, src, bem)

            snr = 3.
            lambda2 = 1. / snr ** 2

            self.stcs[sub_stc_key] = self.apply_inverse_operator_event_wise(epochs[sub_key_epoch], evokeds[sub_key_erp], info, fwd, lambda2, None, 'sLORETA', None)


        if save:
            self.save_event_wise_source_estimates(self.stcs)

        return self.stcs

    def apply_cortical_parcellation_event_stcs(self, stcs, src, save=True, gen_mode=True):

        labels = mne.read_labels_from_annot(self.subject)
        self.labels = [lbl for lbl in labels if lbl.name != 'unknown-lh']
        stc_path = 'stcs/'
        self.stc_cp = dict()

        for key, event_stcs in stcs.items():
            stc_sub_path = stc_path + key + '/'
            event_stcs_cp = np.zeros((68, 500, 5))
            for event_id, event_stc in event_stcs.items():
                event_stc_path = stc_sub_path + event_id + '.csv'
                if gen_mode:
                    label_tc = mne.extract_label_time_course(event_stc, self.labels, src, mode='pca_flip')
                else:
                    label_tc = np.genfromtxt(event_stc_path, delimiter = ',')
                event_stcs_cp[:, :, int(event_id)-1] = label_tc
                if save:
                    np.savetxt(event_stc_path, label_tc, delimiter = ',')
            self.stc_cp[key] = event_stcs_cp
        return self.stc_cp
