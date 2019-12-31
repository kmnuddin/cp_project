#!/usr/bin/env python
# coding: utf-8

# In[1]:


from MNE_Pipeline import MNE_Repo_Mat as MP
import os
import mne
import gc
import re
import pandas as pd


# In[2]:


mne_pp = MP()


# In[3]:


f = mne_pp.load_data('Data/N1.mat')
mne_pp.t.shape


# In[4]:


montage, src, bem = MP.init_exp_for_sl()


# In[5]:


stcs = dict()
stcs_path = 'stcs'
stcs_sub_files = os.listdir(stcs_path)
for stc_sub in stcs_sub_files:
    stcs[stc_sub] = mne_pp.load_stcs(stcs_path, stc_sub)


# In[6]:


stc_cp = mne_pp.apply_cortical_parcellation_event_stcs(stcs, src, save=False, gen_mode=False)


# In[7]:


stc_cp['N1_stc'].shape


# In[8]:


from surfer import Brain

brain = Brain(mne_pp.subject, hemi='both', surf='inflated' ,subjects_dir=mne_pp.subjects_dir, offscreen=False)
brain.add_annotation('aparc')

