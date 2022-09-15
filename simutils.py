import os
import sys
import importlib as imp
from collections import defaultdict

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy
import h5py
import multiprocess as mp

import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import mirnylib.plotting
from cooltools.lib import numutils

import polychrom
from polychrom import polymer_analyses, contactmaps, polymerutils
from polychrom.hdf5_format import list_URIs, load_URI

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 

def Ps_sorter(blocks, bin_edges, chains, cutoff=1.1):
    
    def process(uri):
        idx = int(uri.split('::')[-1])
        try:
            data = load_URI(uri)['pos']
        except:
            print(uri)
        
        ser = {}
        chunk = np.searchsorted(blocks, idx, side='right')
        ser['chunk'] = [chunk]
        
        bins = None
        contacts = None
        for st, end in zip(chains[0:-1],chains[1:]):
            conf = data[st:end,:]
            x,y = polymer_analyses.contact_scaling(conf, bins0=bin_edges, cutoff=cutoff)
            if bins is None:
                bins = x
            if contacts is None:
                contacts = y
            else:
                contact = contacts + y
                
        ser['Ps'] = [(bins, contacts)]
        return pd.DataFrame(ser)
    
    return process

def normalize(x, y, pos):
    
    min_idx = np.argmin(np.abs(x - pos))
    y = y/y[min_idx]
    
    return y


def log_der(x, y, sigma=1):
    logx = np.log10(x)
    logy = np.log10(y)
    
    slope_y = (logy[1:] - logy[0:-1])/(logx[1:] - logx[0:-1])
    slope_y = gaussian_filter1d(slope_y, sigma=1)
    slope_x = 10**((logx[1:] + logx[0:-1])/2)
    
    return slope_x, slope_y


def interp_log_Ps(bin_mids, Ps, N):
    x = np.log10(bin_mids)
    y = np.log10(Ps)
    
    f = interp1d(x, y, kind='linear')

    x1 = np.log10(np.arange(np.floor(bin_mids[-1])))
    y1 = f(x1)
    y1 = np.r_[y1, [y[-1]]*int(N-np.floor(bin_mids[-1]))]
    x1 = np.r_[x1, np.log10(np.arange(np.floor(bin_mids[-1]),N))]

    assert len(x1) == len(y1)
    assert len(x1) == N
    
    return 10**x1, 10**y1


def choose_region(tad_st, tad_end, region_starts, region_ends):
    region_mids = (region_starts+region_ends)//2

    a = np.argmin(np.abs(region_mids - tad_st))
    b = np.argmin(np.abs(region_mids - tad_end))
    
    ind = a
    
    if a != b:
        if (tad_end - region_ends[a]) <= 0:
            ind = a
        elif (tad_st - region_starts[b]) >= 0:
            ind = b
        else:
            return None
    return ind


def tad_metric(pileup):
    assert pileup.shape == (100,100)
    
    inner_slice = (slice(27,47), slice(53,73))
    forward_slice = (slice(51,71), slice(77,97))
    backward_slice = (slice(3,23), slice(29,49))

    mid = np.nanmean(pileup[inner_slice])    
    top = np.nanmean(pileup[forward_slice])
    bot = np.nanmean(pileup[backward_slice])

    out = (top+bot)/2
    
    return mid/out

def dot_metric(pileup):
    assert pileup.shape[0] == pileup.shape[1]
    assert pileup.shape[0] >= 20
    
    N = pileup.shape[0]
    
    w = int(N/20)

    m_peak = int(N/2)
    peak_slice = slice(m_peak - w, m_peak + w), slice(m_peak - w, m_peak + w)
    
    m_bg1 = int(2*N/10)
    bg_slice1 = (slice(m_bg1 - w, m_bg1 + w), slice(m_bg1 - w, m_bg1 + w))
    
    m_bg2 = int(8*N/10)
    bg_slice2 = (slice(m_bg2 - w, m_bg2 + w), slice(m_bg2 - w, m_bg2 + w))
    
    peak = np.nanmean(pileup[peak_slice])    
    bg1 = np.nanmean(pileup[bg_slice1])
    bg2 = np.nanmean(pileup[bg_slice2])

    bg = (bg1 + bg2)/2
    
    return peak/bg