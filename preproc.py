import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
from astropy.io import ascii

# open source module found for converting MJD/JD to DateTimes
import jdutil as jd

# Ben's module -- THANKS BEN
import LCtools

import glob


# LC Time Sort Preprocessing Function

def LC_timesort_preproc(df, resample_len='1d'):
    '''
    Function used to preprocess tabular lightcurve data before inputting them into sonoUno. This is
    done by resampling to a regular time-base and adding NaN values into the observation gaps, so 
    that the timing is correct and silences are observed.

    Inputs ----------------------
        :param df: dataframe with columns mjd, phase, and magnitude
        :type df: pd.Dataframe
        :kwarg resample_len: time-base to resample to. String input that is date-time friendly
        :type resample_len: str
            
    Returns ---------------------
        gaps - preprocessed dataframe. Turn this into a .csv and input into sonoUno
    

    '''
    df['timestamp'] = df.mjd.map(jd.mjd_to_jd).map(jd.jd_to_datetime)
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.set_index('timestamp', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    gaps = df.resample(resample_len).mean()
    gaps.reset_index(inplace=True)
    gaps['mjd_modified'] = gaps.timestamp.map(jd.datetime_to_jd).map(jd.jd_to_mjd)
    gaps = gaps[['mjd_modified', 'mag', 'phase', 'mjd']]
    
    return gaps


from scipy.stats import binned_statistic
from scipy.signal import savgol_filter

# LC Phase Sort Preprocessing Function

def LC_phasesort_preproc(df, bins=None):
    '''
    Function used to preprocess tabular lightcurve data before inputting them into sonoUno. This is
    done by binning the observations by to phase and averaging magnitude observations. This is 
    to ensure that the phase intervals are constant so that the sonoUno sound is played at a regular
    interval.

    Inputs ----------------------
        :param df: dataframe with columns mjd, phase, and magnitude
        :type df: pd.Dataframe
        :kwarg bins: bins to average over
        :type bins: array-like
            
    Returns ---------------------
        new_df - preprocessed dataframe. Turn this into a .csv and input into sonoUno
    

    '''
    df.sort_values('phase', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[['phase', 'mag']]
    # save df cols
    col = df.columns
    # df to numpy
    arr = df.to_numpy()
    # bin by phase
    s, edges, _ = binned_statistic(arr[:,0],arr[:,1], statistic='mean', bins=bins)
    bincenters = edges[:-1]+np.diff(edges)/2
    
    # 2 phases
    bincenters = np.append(bincenters, bincenters+1)
    s = np.append(s, s)

    # back to df - name columns
    new_arr = np.vstack((bincenters, s)).T
    new_df = pd.DataFrame(new_arr, columns = col[0:2])
    
    return new_df




def spectra_preproc(df, bins=None):
    '''
    Function used to preprocess tabular spectral data before inputting them into sonoUno. This is
    done by binning the observations by wavelength and averaging flux observations. This is 
    to ensure that the phase intervals are constant so that the sonoUno sound is played at a regular
    interval.

    Inputs ----------------------
        :param df: dataframe with columns mjd, phase, and magnitude
        :type df: pd.Dataframe
        :kwarg bins: bins to average over
        :type bins: array-like
            
    Returns ---------------------
        new_df - preprocessed dataframe. Turn this into a .csv and input into sonoUno
    

    '''
    df.sort_values('wavelength', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[['wavelength', 'flux']]
    # save df cols
    col = df.columns
    # df to numpy
    arr = df.to_numpy()
    # bin by wavelength
    s, edges, _ = binned_statistic(arr[:,0],arr[:,1], statistic='mean', bins=bins)
    bincenters = edges[:-1]+np.diff(edges)/2
    # back to df - name columns
    new_arr = np.vstack((bincenters, s)).T
    new_df = pd.DataFrame(new_arr, columns = col[0:2])
    
    return new_df


def LC_phasefit_preproc(path_to_lc, bins=None):
    '''
    Function used to fit a phase curve to tabular LC data before inputting them into sonoUno. This is
    done by using a LombScargle periodogram to obtain a period value, then using a Fourier Decomposition
    to make the best fit. 

    Inputs ----------------------
        :param path_to_lc: path to .csv with columns mjd, phase, and magnitude
        :type path_to_lc: str
        :kwarg bins: phase bins to average over
        :type bins: array-like
            
    Returns ---------------------
        df - phase fit dataframe. Turn this into a .csv and input into sonoUno

    '''

    #load data into astropy table, sort by mjd, and remove phase
    data = ascii.read(path_to_lc, format='csv', fast_reader=False)
    data.sort('mjd')
    data.remove_column('phase')
    
    #Use LombScargle periodogram to find the best fit period (in days)
    omega = LCtools.perdiodSearch(data)[0]['P'] 
    
    #restructure data for input to AFD
    data_tuple = (data['mjd'], data['mag'], data['magerr'],)
    
    #Run the Automatic Fourier Decomposition to find the best phase fit
    best_Nterms, best_phase_fit, best_y_fit, best_phased_t, best_resid, best_reduced_ChiSq, best_mtf = LCtools.AFD(data_tuple, omega)
    
    #stack into 2d array
    arr = np.vstack((best_phase_fit, best_y_fit)).T
 
    # bin by phase -> 100 points
    s, edges, _ = binned_statistic(arr[:,0],arr[:,1], statistic='mean', bins=bins)
    bincenters = edges[:-1]+np.diff(edges)/2
    
    # 2 phases
    bincenters = np.append(bincenters, bincenters+1)
    s = np.append(s, s)
    
    #to df - name columns
    new_arr = np.vstack((bincenters, s)).T
    df = pd.DataFrame(new_arr, columns = ['phase', 'mag'])
    
    return df

def phase_bins():
    bins = np.arange(0,1.01,0.01)
    return bins

def wavelength_bins():
    bins = np.arange(3790,7200,10)
    return bins


################## START PREPROCESSING ###################
path_to_data = "TDSS_data/"

#lightcurve paths
lightcurves = glob.glob('{}*LC*.csv'.format(path_to_data))
# remove extras
lightcurves = [ x for x in lightcurves if "Phase" not in x]
lightcurves = [ x for x in lightcurves if "phase" not in x]

#spectra paths
spectra = glob.glob('{}*spec*.csv'.format(path_to_data))

lenpath = len(path_to_data)
#object names
names = [n[lenpath:-9] for n in spectra]

#preprocessing loop
for n in names:
    # Filepaths for each object
    lc_path = [ x for x in lightcurves if n in x][0]
    spec_path = [ x for x in spectra if n in x][0]
    
    #load into pandas dataframes
    lc = pd.read_csv(lc_path)
    spec = pd.read_csv(spec_path)
    
    #do the preprocessing
    time_preproc_lc = LC_timesort_preproc(lc)
    phase_preproc_lc = LC_phasesort_preproc(lc, bins=phase_bins())
    phasefit_preproc_lc = LC_phasefit_preproc(lc_path, bins=phase_bins())
    preproc_spec = spectra_preproc(spec, bins=wavelength_bins())
    
    #save preprocessed files
    time_preproc_lc.to_csv("preproc/{}_LC_timesort.csv".format(n), index=False)
    phase_preproc_lc.to_csv("preproc/{}_LC_phasesort.csv".format(n), index=False)
    phasefit_preproc_lc.to_csv("preproc/{}_LC_phasefit.csv".format(n), index=False)
    preproc_spec.to_csv("preproc/{}_spec.csv".format(n), index=False)

#preprocessed .csvs should be saved in *workingdirectory*/preproc/ folder
