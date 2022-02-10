from numpy.lib.nanfunctions import nanmax
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table
import os

# open source module found for converting MJD/JD to DateTimes
import jdutil as jd

# Ben's module -- THANKS BEN
import LCtools

import glob


# LC Time Sort Preprocessing Function

def LC_timesort_preproc(dataframe, resample_len='1d'):
    '''
    Function used to preprocess tabular light curve data before inputting them into sonoUno. This is
    done by resampling to a regular time-base and adding NaN values into the observation gaps, so 
    that the timing is correct and silences are observed.

    Inputs ----------------------
        :param df: dataframe with columns mjd, phase, and magnitude
        :type df: pd.Dataframe
        :kwarg resample_len: time-base to resample to. String input that is date-time friendly.
        :type resample_len: str
            
    Returns ---------------------
        gaps - preprocessed dataframe. Turn this into a .csv and input into sonoUno
    

    '''
    df = dataframe.copy(deep=True)
    #create timestamp column
    df['timestamp'] = df.mjd.map(jd.mjd_to_jd).map(jd.jd_to_datetime)

    #sort by datetime
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.set_index('timestamp', inplace=True)

    #change timestamp to DatetimeIndex for easy resampling
    df.index = pd.DatetimeIndex(df.index)

    #bin/resample dataframe to chosen time-base 
    gaps = df.resample(resample_len).mean()
    gaps.reset_index(inplace=True)

    #change datetime column back to MJD. Rename columns for input into sonoUno
    gaps['mjd_modified'] = gaps.timestamp.map(jd.datetime_to_jd).map(jd.jd_to_mjd)
    gaps = gaps[['mjd_modified', 'mag', 'mjd']]
    gaps.rename(columns={'mjd_modified':'Modified Julian Day',
                'mag': 'Magnitude'}, inplace=True)
    
    return gaps


from scipy.stats import binned_statistic
from scipy.signal import savgol_filter

# LC Phase Sort Preprocessing Function

def LC_phasesort_preproc(dataframe, bins=None, rephased=False):
    '''
    Function used to preprocess tabular lightcurve data before inputting them into sonoUno. This is
    done by binning the observations by phase and averaging magnitude observations. This is 
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
    df = dataframe.copy(deep=True)
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
    new_df = pd.DataFrame(new_arr, columns = ['Phase', 'Magnitude'])

    #rephase. set this to true if you would like to shift the phase curve 0.5 phases
    if rephased:
        rephase = new_df[new_df.Phase.between(0.505, 1.495)]['Magnitude'].to_numpy()
        rephase = np.append(rephase, rephase)
        rephase_arr = np.vstack((bincenters, rephase)).T
        rephase_df = pd.DataFrame(rephase_arr, columns=['Phase', 'Magnitude'])

        return rephase_df
    
    return new_df


def spectra_preproc(dataframe, bins=None):
    '''
    Function used to preprocess tabular spectral data before inputting them into sonoUno. This is
    done by binning the observations by wavelength and averaging flux observations. This is 
    to ensure that the phase intervals are constant so that the sonoUno sound is played at a regular
    interval.

    Inputs ----------------------
        :param df: dataframe with columns wavelength and flux
        :type df: pd.Dataframe
        :kwarg bins: bins to average over
        :type bins: array-like
            
    Returns ---------------------
        new_df - preprocessed dataframe. Turn this into a .csv and input into sonoUno
    

    '''
    df = dataframe.copy(deep=True)
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
    new_df = pd.DataFrame(new_arr, columns = ['Wavelength', 'Flux'])
    
    return new_df


def LC_phasefit_preproc(path_to_lc, bins=None, rephased=False):
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
    try:
        data.remove_column('phase')
    except:
        pass
    
    flc_data, LC_stat_properties = LCtools.process_LC(data, fltRange=5.0, detrend=True, detrend_deg=2)
    #Use LombScargle periodogram to find the best fit period (in days)
    omega = LCtools.perdiodSearch(flc_data)[0]['P'] 
    print(omega)
    
    #use QualFlag if the data has been processed with process_LC
    try:
        goodQualIndex = np.where(flc_data['QualFlag'] == 1)[0]
        lc_mjd = flc_data['mjd'][goodQualIndex]
        lc_mag = flc_data['mag'][goodQualIndex]
        lc_err = flc_data['magerr'][goodQualIndex]
    except:
        lc_mjd = flc_data['mjd']
        lc_mag = flc_data['mag']
        lc_err = flc_data['magerr']

    #restructure data for input to AFD
    # data_tuple = (flc_data['mjd'], flc_data['mag'], flc_data['magerr'],)
    data_tuple = (lc_mjd, lc_mag, lc_err,)
    
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
    df = pd.DataFrame(new_arr, columns = ['Phase', 'Magnitude'])

    if rephased is True:
        rephase = df[df.Phase.between(0.505, 1.495)]['Magnitude'].to_numpy()
        rephase = np.append(rephase, rephase)
        rephase_arr = np.vstack((bincenters, rephase)).T
        rephase_df = pd.DataFrame(rephase_arr, columns=['Phase', 'Magnitude'])

        return rephase_df
    
    return df


def qualFlagData(flc_data):
    """
    Flexible function used to filter data regardless of whether process_LC has been run yet. Data tuple
    that can be input into AFD is also constructed.
    """
    try:
        goodQualIndex = np.where(flc_data['QualFlag'] == 1)[0]
        lc_mjd = flc_data['mjd'][goodQualIndex]
        lc_mag = flc_data['mag'][goodQualIndex]
        lc_err = flc_data['magerr'][goodQualIndex]
        filtered_data = flc_data[goodQualIndex]
    except:
        lc_mjd = flc_data['mjd']
        lc_mag = flc_data['mag']
        lc_err = flc_data['magerr']
        filtered_data = flc_data

    data_tuple = (lc_mjd, lc_mag, lc_err)

    return filtered_data, data_tuple

def format_df(arr, bins, rephased):
    """
    Function used to format phased LC data into a pandas df that can be input into sonoUno. This means
    binning, averaging, and extending x-axis across 2 phases.
    """
    # bin by phase -> 100 points
    s, edges, _ = binned_statistic(arr[:,0],arr[:,1], statistic='mean', bins=bins)
    bincenters = edges[:-1]+np.diff(edges)/2
    
    # 2 phases
    bincenters = np.append(bincenters, bincenters+1)
    s = np.append(s, s)
    
    #to df - name columns
    new_arr = np.vstack((bincenters, s)).T
    df = pd.DataFrame(new_arr, columns = ['Phase', 'Magnitude'])

    if rephased is True:
        rephase = df[df.Phase.between(0.505, 1.495)]['Magnitude'].to_numpy()
        rephase = np.append(rephase, rephase)
        rephase_arr = np.vstack((bincenters, rephase)).T
        rephase_df = pd.DataFrame(rephase_arr, columns=['Phase', 'Magnitude'])

        return rephase_df
    
    return df


def phase_LC(data, bins=None, rephased=False, **kwargs):
    '''
    Function used to fit a phase curve to tabular LC data before inputting them into sonoUno. This is
    done by using a LombScargle periodogram to obtain a period value, then using a Fourier Decomposition
    to make the best fit. 

    Inputs ----------------------
        :param data: pandas df with columns mjd, mag, magerr, (and optional phase)
        :type data: pandas DataFrame
        :kwarg bins: phase bins to average over
        :type bins: array-like
        :kwarg rephased: if True, move the data over by 0.5 phases
        :type rephased: boolean
            
    Returns ---------------------
        phasefit_df - phase fit dataframe. Turn this into a .csv and input into sonoUno
        phasesort_df - phase sorted dataframe. Turn this into a .csv and input into sonoUno
        period - period in days
    '''
    # option to filter data or not
    # example of a time we wouldn't want to filter would be for phasing data with random flares
    flc = kwargs.pop("flc", False)
    # process_LC inputs
    detrend = kwargs.pop("detrend", False)
    fltRange =  kwargs.pop("fltRange", 5.0)
    detrend_deg = kwargs.pop("detrend_deg", 3)
    Nmax = kwargs.pop("Nmax", 6)
    omega = kwargs.pop("omega", None)

    #pandas to astropy table, to be compatible with Ben's code. Sort by mjd, remove phase if applicable
    lc_table = Table.from_pandas(data)   
    lc_table.sort('mjd')
    try:
        lc_table.remove_column('phase')
    except:
        pass

    #filter data and find period
    if flc == True:
        flc_data, LC_stat_properties = LCtools.process_LC(lc_table, fltRange=fltRange, detrend=detrend, detrend_deg=detrend_deg)
    else:
        flc_data = lc_table

    #Use LombScargle periodogram to find the best fit period (in days)
    filtered_data, data_tuple = qualFlagData(flc_data)

    #determine period, or use input
    if omega == None:
        omega = LCtools.perdiodSearch(filtered_data)[0]['P'] 
    else:
        pass

    #Run the Automatic Fourier Decomposition to find the best phase fit
    best_Nterms, best_phase_fit, best_y_fit, best_phased_t, best_resid, best_reduced_ChiSq, best_mtf = LCtools.AFD(data_tuple, omega, Nmax=Nmax)

    #stack phasefit into 2d array
    phasefit_arr = np.vstack((best_phase_fit, best_y_fit)).T

    #stack phasesort into 2d array
    phasesort_arr = np.vstack((best_phased_t, data_tuple[1])).T

    #restructure to be in sonoUno format
    phasefit_df = format_df(phasefit_arr, bins=bins, rephased=rephased)
    phasesort_df = format_df(phasesort_arr, bins=bins, rephased=rephased)
    
    return omega, phasesort_df, phasefit_df


def phase_bins():
    """
    To be passed as the 'bins' input to any of the phase related functions
    """
    bins = np.arange(0,1.01,0.01)
    return bins

def wavelength_bins():
    """
    To be passed as the 'bins' input to any of the spectra related functions
    """
    bins = np.arange(3790,7200,10)
    return bins



################## PLOTTING ###################

def plot_timesort(raw, preprocessed, **kwargs):
    start = kwargs.pop("start", None)
    end =  kwargs.pop("end", None)

    raw_df = raw.copy(deep=True)
    preproc_df = preprocessed.copy(deep=True)

    if (start is not None) and (end is not None):
        raw_df = raw_df[raw_df.mjd.between(start, end)]
        preproc_df = preproc_df[preproc_df['Modified Julian Day'].between(start, end)]

    fig, ax = plt.subplots()
    ax.scatter(raw_df['mjd'], raw_df['mag'], label='raw', alpha=0.2, color='cornflowerblue')
    ax.scatter(preproc_df['Modified Julian Day'], preproc_df['Magnitude'], 
                alpha=0.2, label='preprocessed', color='salmon')
    ax.invert_yaxis()
    ax.set_xlabel('Modified Julian Day')
    ax.set_ylabel('Magnitude')
    ax.legend()
    
    fig.set_size_inches(7,5)
    fig.set_dpi(120)
    fig.patch.set_facecolor('white')

    return fig


def plot_phased(phasesort_df, phasefit_df):
    fig = plt.figure(figsize=(12,4))
    plt.scatter(phasesort_df['Phase'], phasesort_df['Magnitude'], c='k', marker='o')
    plt.plot(phasefit_df['Phase'], phasefit_df['Magnitude'], 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')
    plt.gca().invert_yaxis()
    plt.title("Phased LC")
    plt.xlabel("Phase")
    plt.ylabel("Magnitude")
    fig.set_size_inches(8, 2.666666666)
    fig.set_dpi(120)
    fig.patch.set_facecolor('white')

    return fig

def plot_spectra(raw, preprocessed, **kwargs):
    raw_df = raw.copy(deep=True)
    preproc_df = preprocessed.copy(deep=True)

    fig, ax = plt.subplots()
    ax.plot(raw_df['wavelength'], raw_df['flux'], label='raw', alpha=0.2, color='cornflowerblue')
    ax.plot(preproc_df['Wavelength'], preproc_df['Flux'], 
                alpha=0.2, label='preprocessed', color='salmon')
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Flux')
    ax.set_xlim(3750, 7250)
    ax.legend()
    
    fig.set_size_inches(7,5)
    fig.set_dpi(120)
    fig.patch.set_facecolor('white')

    return fig