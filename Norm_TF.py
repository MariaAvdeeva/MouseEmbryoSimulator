import numpy as np
import pandas as pd

from scipy.stats import zscore
def zscore_tf(tf, dat):
    zscored = dat.set_index('Branch', append = True).groupby(['stack', 'Time'])['Smooth_'+str(tf)].apply(zscore).droplevel(2)
    dat['Smooth_'+tf+'_zscore'] = zscored.loc[dat.set_index(['Time', 'Branch'], append = True).swaplevel(1,2, axis = 0).index].values



def minmax_tf(tf, dat):
    samples = np.array(dat.index.get_level_values(0).unique())
    max_yap = {}
    min_yap = {}
    for samp in samples:
        rel_dat = dat.loc[samp].query('Gen == 2 or Gen == 3')['Smooth_'+tf]
        max_yap[samp] = max(rel_dat)
        min_yap[samp] = min(rel_dat)
    mmy = np.array([])
    for stack in samples:#range(num_embryos):
        cur_max_yap = max_yap[stack]
        cur_min_yap = min_yap[stack]
        mmy = np.concatenate([mmy, (dat.loc[stack]['Smooth_'+tf].values-cur_min_yap)/(cur_max_yap-cur_min_yap)])
        
    dat['Smooth_'+tf+'_Minmax'] = mmy

def minmax_tf_gen2(tf, dat):
    samples = np.array(dat.index.get_level_values(0).unique())
    max_yap = {}
    min_yap = {}
    for samp in samples:
        rel_dat = dat.loc[samp].query('Gen == 2')['Smooth_'+tf]
        max_yap[samp] = max(rel_dat)
        min_yap[samp] = min(rel_dat)
    mmy = np.array([])
    for stack in samples:#range(num_embryos):
        cur_max_yap = max_yap[stack]
        cur_min_yap = min_yap[stack]
        mmy = np.concatenate([mmy, (dat.loc[stack]['Smooth_'+tf].values-cur_min_yap)/(cur_max_yap-cur_min_yap)])
        
    dat['Smooth_'+tf+'_Minmax_Gen2'] = mmy

def minmax_tf_gen3(tf, dat):
    samples = np.array(dat.index.get_level_values(0).unique())
    max_yap = {}
    min_yap = {}
    for samp in samples:
        rel_dat = dat.loc[samp].query('Gen == 3')['Smooth_'+tf]
        max_yap[samp] = max(rel_dat)
        min_yap[samp] = min(rel_dat)
    mmy = np.array([])
    for stack in samples:#range(num_embryos):
        cur_max_yap = max_yap[stack]
        cur_min_yap = min_yap[stack]
        mmy = np.concatenate([mmy, (dat.loc[stack]['Smooth_'+tf].values-cur_min_yap)/(cur_max_yap-cur_min_yap)])
        
    dat['Smooth_'+tf+'_Minmax_Gen3'] = mmy
    
def mean_tf(tf, dat):
    samples = np.array(dat.index.get_level_values(0).unique())
    mean_yap = {}
    for samp in samples:
        rel_dat = dat.loc[samp].query('Gen == 2 or Gen == 3')['Smooth_'+tf]
        mean_yap[samp] = np.mean(rel_dat)
    my = np.array([])
    for stack in samples:#range(num_embryos):
        cur_mean_yap = mean_yap[stack]
        my = np.concatenate([my, (dat.loc[stack]['Smooth_'+tf].values)/(cur_mean_yap)])

    dat['Smooth_'+tf+'_Mean'] = my

def mean_tf_gen2(tf, dat):
    samples = np.array(dat.index.get_level_values(0).unique())
    mean_yap = {}
    for samp in samples:
        rel_dat = dat.loc[samp].query('Gen == 2')['Smooth_'+tf]
        mean_yap[samp] = np.mean(rel_dat)
    my = np.array([])
    for stack in samples:#range(num_embryos):
        cur_mean_yap = mean_yap[stack]
        my = np.concatenate([my, (dat.loc[stack]['Smooth_'+tf].values)/(cur_mean_yap)])

    dat['Smooth_'+tf+'_Mean_Gen2'] = my

def mean_tf_gen3(tf, dat):
    samples = np.array(dat.index.get_level_values(0).unique())
    mean_yap = {}
    for samp in samples:
        rel_dat = dat.loc[samp].query('Gen == 3')['Smooth_'+tf]
        mean_yap[samp] = np.mean(rel_dat)
    my = np.array([])
    for stack in samples:#range(num_embryos):
        cur_mean_yap = mean_yap[stack]
        my = np.concatenate([my, (dat.loc[stack]['Smooth_'+tf].values)/(cur_mean_yap)])

    dat['Smooth_'+tf+'_Mean_Gen3'] = my