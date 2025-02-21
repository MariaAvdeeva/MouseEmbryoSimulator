import numpy as np
import pandas as pd

# warp time
def warp_time(dat):
    # warp time
    dts = {}
    dat = dat.sort_values('Time')
    for stack in np.unique(dat.index.get_level_values(0)):
        dts[stack] = []
        dt = dat.loc[stack]
        for sm_br in np.unique(dt['SmallBranch']):
            cur_gen = int(len(sm_br.split('_'))-1)
            cur_dt = dt.loc[dt['SmallBranch'] == sm_br]
            min_time = min(cur_dt['Time'])
            max_time = max(cur_dt['Time'])+1
            cur_dt.loc[:, 'Warped_time'] = (cur_gen + (cur_dt['Time']-min_time)/(max_time-min_time)).values
            cur_dt.loc[:,'Warped_time'] = (0.05*(cur_dt['Warped_time'].values/0.05).astype(int))
            dts[stack].append(cur_dt)
        dts[stack] = pd.concat(dts[stack])
    dts = pd.concat(dts)
    dat = dts
    return dat

def gen16_time(dat):
    # define within cycle time
    dts = {}
    dat = dat.sort_values('Time')
    for stack in np.unique(dat.index.get_level_values(0)):
        dts[stack] = []
        dt = dat.loc[stack]
        for sm_br in np.unique(dt['Branch']):
            cur_gen = int(len(sm_br.split('_'))-1)
            cur_dt = dt.loc[dt['Branch'] == sm_br]
            try:
                min_time = cur_dt.query('Gen == 1').sort_values('Time').iloc[-1,:]['Time']
            except:
                min_time = 0
            cur_dt.loc[:,'Gen16_time'] = cur_dt['Time']-min_time
            dts[stack].append(cur_dt)
        dts[stack] = pd.concat(dts[stack])
    dts = pd.concat(dts)
    dat = dts
    dat['Aligned_16cs_Hour'] = dat['Gen16_time']/4
    return dat

def gen32_time(dat):
    # define within cycle time
    dts = {}
    dat = dat.sort_values('Time')
    for stack in np.unique(dat.index.get_level_values(0)):
        dts[stack] = []
        dt = dat.loc[stack]
        for sm_br in np.unique(dt['Branch']):
            cur_gen = int(len(sm_br.split('_'))-1)
            cur_dt = dt.loc[dt['Branch'] == sm_br]
            try:
                min_time = cur_dt.query('Gen == 2').sort_values('Time').iloc[-1,:]['Time']
            except:
                min_time = 0
            cur_dt.loc[:,'Gen32_time'] = cur_dt['Time']-min_time
            dts[stack].append(cur_dt)
        dts[stack] = pd.concat(dts[stack])
    dts = pd.concat(dts)
    dat = dts
    dat['Aligned_32cs_Hour'] = dat['Gen32_time']/4
    return dat