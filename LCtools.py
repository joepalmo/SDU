import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

from scipy.stats import f
from scipy.stats import skew

import astropy.units as u
from astropy.timeseries import LombScargle
from astropy.table import Table
from astropy.io import votable as vo

import warnings


__all__ = ['AFD', 'bin_phaseLC', 'get_ZTF_LC', 'MultiTermFit',
           'perdiodSearch', 'plt_lc', 'process_LC']


@u.quantity_input(ra='angle', dec='angle', radius='angle')
def get_ZTF_LC(ra, dec, radius=3*u.arcsec, ZTFfilter=None):
    ra = ra.to(u.deg)
    dec = dec.to(u.deg)
    str_r = str(np.round(radius.to(u.deg).value, 6))

    if ZTFfilter:
        LC_url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE%20{ra.value}%20{dec.value}%20{str_r}BANDNAME={ZTFfilter}"
    else:
        LC_url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE%20{ra.value}%20{dec.value}%20{str_r}"
    
    LC = vo.parse(LC_url)
    LC = LC.get_first_table().to_table(use_names_over_ids=True)

    return LC


class MultiTermFit:
    """Multi-term Fourier fit to a light curve
    Parameters
    ----------
    omega : float
        angular frequency of the fundamental mode
    n_terms : int
        the number of Fourier modes to use in the fit
    """
    def __init__(self, omega, n_terms):
        self.omega = omega
        self.n_terms = n_terms

    def _make_X(self, t):
        t = np.asarray(t)
        k = np.arange(1, self.n_terms + 1)
        X = np.hstack([np.ones(t[:, None].shape),
                       np.sin(k * self.omega * t[:, None]),
                       np.cos(k * self.omega * t[:, None])])
        return X

    def fit(self, t, y, dy):
        """Fit multiple Fourier terms to the data
        Parameters
        ----------
        t: array_like
            observed times
        y: array_like
            observed fluxes or magnitudes
        dy: array_like
            observed errors on y
        Returns
        -------
        self :
            The MultiTermFit object is  returned
        """
        t = np.asarray(t)
        y = np.asarray(y)
        dy = np.asarray(dy)

        self.y_ = y
        self.dy_ = dy
        self.t_ = t

        self.C_ = np.diag(dy * dy)

        X_scaled = self._make_X(t) / dy[:, None]
        y_scaled = y / dy

        self.w_ = np.linalg.solve(np.dot(X_scaled.T, X_scaled),
                                  np.dot(X_scaled.T, y_scaled))
        self.cov_ = np.linalg.inv(np.dot(X_scaled.T, X_scaled))
        return self

    def predict(self, Nphase, return_phased_times=False, adjust_offset=True):
        """Compute the phased fit, and optionally return phased times
        Parameters
        ----------
        Nphase : int
            Number of terms to use in the phased fit
        return_phased_times : bool
            If True, then return a phased version of the input times
        adjust_offset : bool
            If true, then shift results so that the minimum value is at phase 0
        Returns
        -------
        phase, y_fit : ndarrays
            The phase and y value of the best-fit light curve
        phased_times : ndarray
            The phased version of the training times.  Returned if
            return_phased_times is set to  True.
        """
        phase_fit = np.linspace(0, 1, Nphase + 1)[:-1]

        X_fit = self._make_X(2 * np.pi * phase_fit / self.omega)
        y_fit = np.dot(X_fit, self.w_)
        i_offset = np.argmin(y_fit)

        if adjust_offset:
            y_fit = np.concatenate([y_fit[i_offset:], y_fit[:i_offset]])

        if return_phased_times:
            if adjust_offset:
                offset = phase_fit[i_offset]
            else:
                offset = 0
            phased_times = (self.t_ * self.omega * 0.5 / np.pi - offset) % 1

            return phase_fit, y_fit, phased_times

        else:
            return phase_fit, y_fit

    def modelfit(self, time):
        y_t = self.w_[0] 
        for ii in range(self.n_terms):
            coefs = self.w_[2*ii+1:2*ii+3]
            y_t += coefs[0]*np.sin((ii+1)*self.omega*time) + coefs[1]*np.cos((ii+1)*self.omega*time)
        return y_t


def _oldAFD(data, period, alpha=0.99, Nmax=6):
    omega = (2.0 * np.pi) / period
    mjd, mag, err = data

    add_another_term = True
    Nterms1 = 1
    Nterms2 = 2
    while add_another_term:
        n_fit_params1 = (Nterms1 * 2) + 1
        n_fit_params2 = (Nterms2 * 2) + 1

        mtf1 = MultiTermFit(omega, Nterms1)
        mtf1.fit(mjd, mag, err)
        phase_fit1, y_fit1, phased_t1 = mtf1.predict(1000, return_phased_times=True, adjust_offset=True)
        inter_y_fit1 = np.interp(phased_t1, phase_fit1, y_fit1)

        resid1 = mag - inter_y_fit1
        ChiSq1 = np.sum((resid1 / err)**2)
        reduced_ChiSq1 = ChiSq1 / (resid1.size - n_fit_params1)

        mtf2 = MultiTermFit(omega, Nterms2)
        mtf2.fit(mjd, mag, err)
        phase_fit2, y_fit2, phased_t2 = mtf2.predict(1000, return_phased_times=True, adjust_offset=True)
        inter_y_fit2 = np.interp(phased_t2, phase_fit2, y_fit2)

        resid2 = mag - inter_y_fit2
        ChiSq2 = np.sum((resid2 / err)**2)
        reduced_ChiSq2 = ChiSq2 / (resid2.size - n_fit_params2)

        b = resid2.size - n_fit_params2
        a = n_fit_params2 - n_fit_params1

        fstat = (ChiSq1 / ChiSq2 - 1) * (b / a)
        # fstat = (reduced_ChiSq1 / reduced_ChiSq2 - 1) * (b / a)

        df1 = a
        df2 = b
        test_value = f.ppf(alpha, dfn=df1, dfd=df2)

        if fstat <= test_value:
            add_another_term = False
        elif Nterms1 == Nmax:
            add_another_term = False
        else:
            Nterms1 += 1
            Nterms2 += 1

    return Nterms1, phase_fit1, y_fit1, phased_t1, resid1, reduced_ChiSq1, mtf1


def AFD(data, period, alpha=0.99, Nmax=6):
    """Automatic Fourier Decomposition
    """
    omega = (2.0 * np.pi) / period
    mjd, mag, err = data

    add_another_term = True
    #alpha = 0.99
    Nterms1 = 1
    while add_another_term:
        fstats = np.zeros(2)
        test_values = np.zeros(2)

        if Nterms1 >= Nmax:
            add_another_term = False
            best_Nterms = Nmax

        for Nterms2 in np.arange(Nterms1 + 1, Nterms1 + 3):
            n_fit_params1 = (Nterms1 * 2) + 1
            n_fit_params2 = (Nterms2 * 2) + 1

            mtf1 = MultiTermFit(omega, Nterms1)
            mtf1.fit(mjd, mag, err)
            phase_fit1, y_fit1, phased_t1 = mtf1.predict(1000, return_phased_times=True, adjust_offset=True)
            inter_y_fit1 = np.interp(phased_t1, phase_fit1, y_fit1)

            resid1 = mag - inter_y_fit1
            ChiSq1 = np.sum((resid1 / err)**2)
            reduced_ChiSq1 = ChiSq1 / (resid1.size - n_fit_params1)

            mtf2 = MultiTermFit(omega, Nterms2)
            mtf2.fit(mjd, mag, err)
            phase_fit2, y_fit2, phased_t2 = mtf2.predict(1000, return_phased_times=True, adjust_offset=True)
            inter_y_fit2 = np.interp(phased_t2, phase_fit2, y_fit2)

            resid2 = mag - inter_y_fit2
            ChiSq2 = np.sum((resid2 / err)**2)
            reduced_ChiSq2 = ChiSq2 / (resid2.size - n_fit_params2)

            b = resid2.size - n_fit_params2
            a = n_fit_params2 - n_fit_params1

            # fstat = (ChiSq1 / ChiSq2 - 1) * (b / a)
            fstats[Nterms2 - Nterms1 - 1] = (reduced_ChiSq1 / reduced_ChiSq2 - 1) * (b / a)

            df1 = resid1.size - n_fit_params1  # a
            df2 = resid2.size - n_fit_params2  # b
            test_values[Nterms2 - Nterms1 - 1] = f.ppf(alpha, dfn=df1, dfd=df2)

        if (fstats[0] <= test_values[0]) & (fstats[1] <= test_values[1]):
            add_another_term = False
            best_Nterms = Nterms1
        elif (fstats[0] <= test_values[0]) & ((fstats[0] > test_values[0])):
            Nterms1 += 2
        else:
            Nterms1 += 1

    best_mtf = MultiTermFit(omega, best_Nterms)
    best_mtf.fit(mjd, mag, err)
    best_phase_fit, best_y_fit, best_phased_t = best_mtf.predict(1000, return_phased_times=True, adjust_offset=True)
    best_inter_y_fit = np.interp(best_phased_t, best_phase_fit, best_y_fit)

    best_resid = mag - best_inter_y_fit
    best_ChiSq = np.sum((best_resid / err)**2)
    best_reduced_ChiSq = best_ChiSq / (best_resid.size - n_fit_params2)

    return best_Nterms, best_phase_fit, best_y_fit, best_phased_t, best_resid, best_reduced_ChiSq, best_mtf


def plt_lc(lc_data, all_period_properties, ax=None, fig=None, title="", plt_resid=False, phasebin=False, bins=25, RV=False, phasebinonly=False, show_err_lines=True, plot_rejected=False):
    goodQualIndex = np.where(lc_data['QualFlag']==True)[0]
    badQualIndex = np.where(lc_data['QualFlag']==False)[0]
    mjd = lc_data['mjd'][goodQualIndex].data
    mag = lc_data['mag'][goodQualIndex].data
    err = lc_data['magerr'][goodQualIndex].data

    AFD_data = all_period_properties['AFD']
    if len(AFD_data)==8:
        deasliased_period, Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiS, mfit = AFD_data
    else:
        Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiS, mfit = AFD_data

    t0 = all_period_properties['t0']
    P = all_period_properties['P']
    mjd_bad = lc_data['mjd'][badQualIndex].data
    phase_bad = ((mjd_bad - t0) / P) % 1
    mag_bad = lc_data['mag'][badQualIndex].data
    err_bad = lc_data['magerr'][badQualIndex].data

    binned_phase, binned_mag, binned_err = bin_phaseLC(phased_t, mag, err, bins=bins)

    is_Periodic = all_period_properties['is_Periodic']
    if  is_Periodic:
        if plt_resid:
            frame1=fig.add_axes((.1,.3,.8,.6))
            #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
            plt_ax = frame1
        else:
            plt_ax = plt.gca()
        
        plt_ax.set_title(title)
        #plt_ax.title("Nterms: "+str(Nterms))
        if phasebinonly:
            pass
        else:
            if RV:
                plt_ax.errorbar(phased_t, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
                plt_ax.errorbar(phased_t+1, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
            else:
                plt_ax.errorbar(phased_t, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
                plt_ax.errorbar(phased_t+1, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)

                if plot_rejected:
                    plt_ax.errorbar(phase_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)
                    plt_ax.errorbar(phase_bad+1, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)
            #plt_ax.errorbar(phased_t+2, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)

            # N_terms
        plt_ax.plot(phase_fit, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')
        plt_ax.plot(phase_fit+1, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')
        #plt_ax.plot(phase_fit+2, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        
        plt_ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        if show_err_lines:
            plt_ax.axhline(fmagmn+3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5, label='3X Mag Err')
            plt_ax.axhline(fmagmn-3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5)
            plt_ax.axhline(fmagmn+3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5, label='3X Mag StDev')
            plt_ax.axhline(fmagmn-3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5)
        else:
            pass
        #plt_ax.set_xlim(0.0, 2.0)
        #plt_ax.set_ylim(18.2, 16.7)
        if RV:
            plt_ax.set_ylabel('RV [km s$^{-1}$]')
        else:
            plt_ax.set_ylabel('mag')
            #plt_ax.set_xlabel('Phase')
            plt_ax.invert_yaxis()

        plt_ax.grid()
        if phasebin:
            plt_ax.errorbar(binned_phase, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=1.5, alpha=0.3)
            plt_ax.errorbar(binned_phase+1, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=1.5, alpha=0.3)

        if plt_resid:
            frame1.set_xlim(0.0, 2.0)
            frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
            frame2 = fig.add_axes((.1,.1,.8,.2))

            frame2.errorbar(phased_t, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=0.3)
            frame2.errorbar(phased_t+1, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=0.3)
            #plt_ax.errorbar(phased_t+2, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=0.3)

            frame2.grid()
            #ax2 = plt.gca()
            frame2.set_xlim(0.0, 2.0)
            frame2.set_ylim(1.5*resid.min(), 1.5*resid.max())
            #frame2.yaxis.set_major_locator(plt.MaxNLocator(4))

            frame2.set_xlabel(r'Phase')
            frame2.set_ylabel('Residual')# \n N$_{terms} = 4$')
    else:
        #fig1 = plt.figure()
        plt_ax = plt.gca()
        plt_ax.set_title(title)
        if RV:
            plt_ax.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
        else:
            plt_ax.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
            if plot_rejected:
                plt_ax.errorbar(mjd_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        plt_ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        plt_ax.axhline(fmagmn+3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5, label='3X Mag Err')
        plt_ax.axhline(fmagmn-3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5)
        plt_ax.axhline(fmagmn+3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5, label='3X Mag StDev')
        plt_ax.axhline(fmagmn-3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5)

        #plt_ax.set_xlim(0.0, 2.0)
        #plt_ax.set_ylim(18.2, 16.7)
        plt_ax.set_xlabel('MJD')
        plt_ax.grid()
        if RV:
            plt_ax.set_ylabel('RV [km s$^{-1}$]')
        else:
            plt_ax.set_ylabel('mag')
            plt_ax.invert_yaxis()


def bin_phaseLC(phase, mag, err, bins=25):
    binned_phase = np.linspace(start=0, stop=1.0, num=bins)
    N = binned_phase.size
    phase_bin_delta = (binned_phase[1] - binned_phase[0])/2.

    binned_mag = np.empty(N) * np.nan
    binned_err = np.empty(N) * np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for ii in range(N):
            index = np.where((phase >= binned_phase[ii]-phase_bin_delta) & (phase < binned_phase[ii]+phase_bin_delta))[0]
            binned_mag[ii] = np.mean(mag[index])
            binned_err[ii] = np.sqrt((err[index]**2).sum())/index.size

    return binned_phase, binned_mag, binned_err


def process_LC(lc_data, fltRange=5.0, detrend=False, detrend_deg=3):
    lc_mjd = lc_data['mjd']
    lc_mag = lc_data['mag']
    lc_err = lc_data['magerr']

    Tspan100 = lc_mjd.max() - lc_mjd.min()
    Tspan95 = np.percentile(lc_mjd, 100 - fltRange) - np.percentile(lc_mjd, fltRange)

    nmag = len(lc_mag)

    # magmn = np.mean(lc_mag)
    errmn = np.mean(lc_err)

    brt10per = np.percentile(lc_mag, fltRange)
    fnt10per = np.percentile(lc_mag, 100 - fltRange)
    brt10data = lc_data[lc_mag <= brt10per]
    fnt10data = lc_data[lc_mag >= fnt10per]

    medmagfnt10 = np.median(fnt10data['mag'])
    mederrfnt10 = np.median(fnt10data['magerr'])

    medmagbrt10 = np.median(brt10data['mag'])
    mederrbrt10 = np.median(brt10data['magerr'])

    brtcutoff = medmagbrt10 - (2 * mederrbrt10)
    fntcutoff = medmagfnt10 + (2 * mederrfnt10)

    # filter_data = lc_data[(lc_mag >= brtcutoff) & (lc_mjd >= 0.0)]
    # flc_data = filter_data[filter_data['mag'] <= fntcutoff]


    filter_index = (lc_mag >= brtcutoff) & (lc_mag <= fntcutoff) & (lc_mjd >= 0.0)
    lc_data.add_column(filter_index, name='QualFlag')

    flc_data = lc_data[filter_index]

    # flc_data = flc_data[flc_data['magerr'] > 0.05]

    flc_mag = flc_data['mag']
    flc_mjd = flc_data['mjd']
    flc_err = flc_data['magerr']
    flc_nmag = len(flc_mag)

    if flc_nmag >= 10:
        p_lin, residuals_lin, rank_lin, singular_values_lin, rcond_lin = np.polyfit(flc_mjd-flc_mjd.min(), flc_mag, 1, rcond=None, full=True, w=1.0/flc_err, cov=False)  # linear fit to LC
        p_quad, residuals_quad , rank_quad, singular_values_quad, rcond_quad = np.polyfit(flc_mjd-flc_mjd.min(), flc_mag, 2, rcond=None, full=True, w=1.0/flc_err, cov=False)  # Quadratic fit to LC
        residuals_lin *= (1. / (flc_nmag - 1))
        residuals_quad *= (1. / (flc_nmag - 1))
    else:
        p_lin = np.array([0, 0])
        residuals_lin = np.array([99999.9999])
        p_quad = np.array([0, 0, 0])
        residuals_quad = np.array([99999.9999])

    mean_error = np.nanmean(flc_err)
    lc_std = np.nanstd(flc_mag)
    var_stat = lc_std / mean_error

    con = conStat(flc_mag)

    fmagmed = np.median(flc_mag)
    fmagmn = np.mean(flc_mag)
    fmagmax = np.max(flc_mag)
    fmagmin = np.min(flc_mag)

    # ferrmed = np.median(flc_err)
    ferrmn = np.mean(flc_err)
    # fmag_stdev = np.std(flc_mag)

    rejects = nmag - flc_nmag

    mag_above = np.mean(flc_mag) - (2 * ferrmn)
    mag_below = np.mean(flc_mag) + (2 * ferrmn)

    nabove = np.where(flc_mag <= mag_above)[0].size
    nbelow = np.where(flc_mag >= mag_below)[0].size

    Mt = (fmagmax - fmagmed) / (fmagmax - fmagmin)

    minpercent = np.percentile(flc_mag, 5)
    maxpercent = np.percentile(flc_mag, 95)

    a95 = maxpercent - minpercent

    lc_skew = skew(flc_mag)

    summation_eqn1 = np.sum(((flc_mag - fmagmn)**2) / (flc_err**2))
    Chi2 = (1. / (flc_nmag - 1)) * summation_eqn1

    properties = {'Tspan100':Tspan100, 'Tspan95':Tspan95, 'a95':a95, 'Mt':Mt, 'lc_skew':lc_skew,
                  'Chi2':Chi2, 'brtcutoff':brtcutoff, 'brt10per':brt10per, 'fnt10per':fnt10per,
                  'fntcutoff':fntcutoff, 'errmn':errmn, 'ferrmn':ferrmn, 'ngood':flc_nmag,
                  'nrejects':rejects, 'nabove':nabove, 'nbelow':nbelow, 'VarStat':var_stat,
                  'Con':con, 'm':p_lin[0], 'b_lin':p_lin[1], 'chi2_lin':residuals_lin[0],
                  'a':p_quad[0], 'b_quad':p_quad[1], 'c':p_quad[2], 'chi2_quad':residuals_quad[0]}

    # flc_data = [flc_mjd, flc_mag, flc_err]
    # return Table(flc_data), properties
    if detrend:
        trend_fit = np.polyfit(lc_data['mjd'], lc_data['mag'], deg=detrend_deg, w=1 / lc_data['magerr'])
        rescaled_mag = (lc_data['mag'] / np.polyval(trend_fit, lc_data['mjd'])) * np.nanmean(lc_data['mag'])
        lc_data['mag'] = rescaled_mag

    return lc_data, properties


def conStat(values, threshold=2., n_con=3):
    '''
    See Wozniak 2000.
    Return Wozniak index = (number of consecutive series / (N - 2))
    See Shin et al. 2009 as well.

    values : list of magnitudes.

    threshold : limit of consecutive points.

    n_con : miminum number of consecutive points to generate a consecutive series.
    '''

    median_value = np.median(values)
    std_value = np.std(values)
    n_value = len(values)

    #remember that this is magnitude based values.
    bright_index = np.where(values <= median_value - threshold * std_value)[0]
    faint_index = np.where(values >= median_value + threshold * std_value)[0]
    if n_value <= 2:
            return 0.

    def find_con_series(indices):
            n_con = 0
            #pre_i = -1
            for i in range(1, len(indices) - 1):
                    if indices[i] == (indices[i - 1] + indices[i + 1]) / 2:
                            #if i != pre_i + 1:
                            n_con += 1
                            #pre_i = i

                            #count all the lowest points of the consecutive indices
            for i in range(2, len(indices) - 1):
                    if (indices[i] == (indices[i - 1] + indices[i + 1]) / 2) and (indices[i - 1] != (indices[i - 2] + indices[i]) / 2):
                            n_con += 1


                            #count all the highest points of the consecutive indices
            for i in range(1, len(indices) - 2):
                    if (indices[i] == (indices[i - 1] + indices[i + 1]) / 2) and (indices[i + 1] != (indices[i + 2] + indices[i]) / 2):
                            n_con += 1

            return n_con

    return (find_con_series(bright_index)  + find_con_series(faint_index)) / float(n_value - 2)


def chisq(data, model):
    return np.sum(((data - model) / data)**2)


def perdiodSearch(lc_data, minP=0.1, maxP=100.0, alias_checks=[np.log10(0.5), np.log10(1.0), np.log10(3.0), np.log10(29.5306)], whitten=True, log10FAP=-10.0, aliasSampleRegion=0.05, checkHarmonic=True):
    
    lc_mjd = lc_data['mjd']
    lc_mag = lc_data['mag']
    lc_err = lc_data['magerr']

    # assume all data is quality
    #goodQualIndex = np.where(lc_data['QualFlag'] == 1)[0]
    #lc_mjd = lc_data['mjd'][goodQualIndex]
    #lc_mag = lc_data['mag'][goodQualIndex]
    #lc_err = lc_data['magerr'][goodQualIndex]

    t_days = lc_mjd  # * u.day
    y_mags = lc_mag  # * u.mag
    dy_mags = lc_err  # * u.mag

    if t_days.unit == None:
        t_days = t_days * u.d
    if y_mags.unit == None:
        y_mags = y_mags * u.mag
    if dy_mags.unit == None:
        dy_mags = dy_mags * u.mag

    # maxP = 1000.0 * u.d
    minP = minP * u.d  # 0.1 * u.d

    maximum_frequency = (minP)**-1
    # minimum_frequency = (maxP)**-1

    frequency = np.linspace(0, maximum_frequency.value, num=250001)[1:] / u.d

    ls = LombScargle(t_days, y_mags, dy_mags, fit_mean=True, center_data=True)
    power = ls.power(frequency=frequency)

    top_N_periods = power.size
    top_frequency_powers_index = np.argsort(power)[-top_N_periods:][::-1]
    top_frequency = frequency[top_frequency_powers_index]
    top_period = 1 / top_frequency
    log10_P = np.log10(top_period.value)

    sample_around_logP_region = aliasSampleRegion
    is_alias = np.zeros(log10_P.size, dtype='int')
    for aliasP in alias_checks:
        is_alias += ((log10_P >= aliasP - sample_around_logP_region) & (log10_P <= aliasP + sample_around_logP_region)).astype(int)

    is_alias = is_alias.astype(bool)

    if whitten:
        time_whittened = 0
        data = [lc_mjd, lc_mag, lc_err]
        still_aliased = is_alias[0]
        while still_aliased:
            best_period = top_period.value[0]
            AFD_data = AFD(data, best_period, Nmax=1)
            Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiSq, mfit = AFD_data

            ls = LombScargle(t_days, resid.data * u.mag, dy_mags)
            power = ls.power(frequency=frequency)

            top_N_periods = power.size
            top_frequency_powers_index = np.argsort(power)[-top_N_periods:][::-1]

            top_frequency = frequency[top_frequency_powers_index]
            top_period = 1 / top_frequency

            log10_P = np.log10(top_period.value)

            sample_around_logP_region = aliasSampleRegion
            is_alias = np.zeros(log10_P.size, dtype='int')
            for aliasP in alias_checks:
                is_alias += ((log10_P >= aliasP - sample_around_logP_region) & (log10_P <= aliasP + sample_around_logP_region)).astype(int)

            still_aliased = is_alias[0]
            data = [lc_mjd, resid, lc_err]
            time_whittened += 1

    first_non_alias = np.where(~is_alias)[0][0]
    isAlias = time_whittened > 0

    best_period = top_period[first_non_alias].value
    best_period_FAP = np.log10(ls.false_alarm_probability(ls.power((best_period * u.d)**-1)))
    if checkHarmonic:
        P0 = best_period
        harmonics = createHarmonicFrac(Nmax=4)
        lc_data = [t_days.value, y_mags.value, dy_mags.value]
        best_period, best_harmonic, best_resids = checkHarmonics(P0, harmonics, lc_data, nterms=6)

    data = [lc_mjd, lc_mag, lc_err]
    AFD_data = AFD(data, best_period)
    Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiSq, mfit = AFD_data
    Amp = y_fit.max() - y_fit.min()
    t0 = (lc_mjd - (phased_t * best_period)).max()

    omega_best = (2.0 * np.pi) / best_period
    is_Periodic = best_period_FAP <= log10FAP

    all_properties = {'P': best_period, 'omega_best': omega_best, 'is_Periodic': is_Periodic,
                      'logProb': best_period_FAP, 'Amp': Amp, 'isAlias': isAlias,
                      'time_whittened': time_whittened, 'ls': ls, 'frequency': frequency, 'power': power,
                      'minP': minP, 'maxP': maxP, 'AFD': AFD_data, 't0': t0}

    select_properties = {'P': best_period, 'logProb': best_period_FAP, 'Amp': Amp,
                         'isAlias': isAlias, 'time_whittened': time_whittened}

    return select_properties, all_properties


def createHarmonicFrac(Nmax=4):
    hold = np.arange(2.0, Nmax + 1.0)
    return np.sort(np.hstack(([1], hold, hold**-1)))


def checkHarmonics(P0, harmonics, lc_data, nterms=6):
    # P0 = P0 * u.d
    ChiSqs = np.zeros(harmonics.size) * np.nan
    for ii, harm in enumerate(harmonics):
        P_test = harm * P0
        AFD_data = AFD(lc_data, P_test, Nmax=nterms)
        Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiSq, mfit = AFD_data
        ChiSqs[ii] =  reduced_ChiSq

    best_harmonic = harmonics[np.nanargmin(ChiSqs)]
    P = P0 * best_harmonic
    return P, best_harmonic, ChiSqs


def dealias_Pal2013(P, A, skew, gmi):
    if (-0.55 <= np.log10(A) <= +0.30) & (-1.20 <= skew <= +0.20) & (-0.42 <= gmi <= +0.50):  # (RRab)
        if (-0.36 <= np.log10(P) <= -0.05):
            true_P = P
        elif (np.log10(P) < -0.36):
            true_P = P * 2
        elif (np.log10(P) > -0.05):
            true_P = P * 0.5
        VarType = 'RRab'
        return true_P, VarType
    if (-0.55 <= np.log10(A) <= -0.15) & (-0.40 <= skew <= +0.35) & (-0.20 <= gmi <= +0.35):  # (RRc)
        if (-0.59 <= np.log10(P) <= -0.36):
            true_P = P
            VarType = 'RRc'
            return true_P, VarType
    if (-0.7 <= np.log10(A) <= +0.00) & (0.32 <= skew <= 3.6) & (-0.2 <= gmi <= 3.0):  # (Single min)
        if (-0.6 <= np.log10(P)):
            true_P = P
            VarType = 'SingleMin'
            return true_P, VarType
    if (-0.67 <= np.log10(A) <= 0.14) & (+1.00 <= skew <= 3.70) & (-1.20 <= gmi <= 3.80):  # Algol (EA)
        if (-0.6 <= np.log10(P)):
            true_P = P
            VarType = 'EA'
            return true_P, VarType
        elif (-0.6 <= np.log10(2*P)):
            true_P = 2*P
            VarType = "EA"
            return true_P, VarType
    if (-0.56 <= np.log10(A) <= -0.09) & (-0.10 <= skew <= +1.60) & (+0.10 <= gmi <= +1.80):  # β Lyr and W UMa (EB/EW)
        if (-0.67 <= np.log10(P) <= -0.40):
            true_P = P
            VarType = 'EB/EW'
            return true_P, VarType
        elif (-0.67 <= np.log10(2 * P) <= -0.40):
            true_P = 2 * P
            VarType = 'EB/EW'
            return true_P, VarType
    if (-0.63 <= np.log10(A) <= -0.12) & (-1.00 <= skew <= +0.70) & (-0.50 <= gmi <= +0.20):  # SX Phe/δ Sct (deltScut)
        if (-1.38 <= np.log10(P) <= -1.05):
            true_P = P
            VarType = 'SXPhe/deltScut'
            return true_P, VarType

    true_P = None
    VarType = "Unknown"
    return true_P, VarType


def plt_any_lc(lc_data, P, is_Periodic=False, ax=None, fig=None, title="", plt_resid=False, phasebin=False, bins=25, RV=False, phasebinonly=False, show_err_lines=True, plot_rejected=False):
    goodQualIndex = np.where(lc_data['QualFlag']==True)[0]
    badQualIndex = np.where(lc_data['QualFlag']==False)[0]
    mjd = lc_data['mjd'][goodQualIndex].data
    mag = lc_data['mag'][goodQualIndex].data
    err = lc_data['magerr'][goodQualIndex].data

    data = [mjd, mag, err]
    AFD_data = AFD(data, P)

    Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiS, mfit = AFD_data

    Amp = y_fit.max() - y_fit.min()
    t0 = (mjd - (phased_t * P)).min()

    title = title  # + "Amp = {!s} $|$ t0 = {!s}".format(np.round(Amp, 3), np.round(t0, 7))

    mjd_bad = lc_data['mjd'][badQualIndex].data
    phase_bad = ((mjd_bad - t0) / P) % 1
    mag_bad = lc_data['mag'][badQualIndex].data
    err_bad = lc_data['magerr'][badQualIndex].data

    binned_phase, binned_mag, binned_err = bin_phaseLC(phased_t, mag, err, bins=bins)

    # is_Periodic = True
    if is_Periodic:
        if plt_resid:
            frame1 = fig.add_axes((.1, .3, .8, .6))
            # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
            plt_ax = frame1
        else:
            plt_ax = plt.gca()

        plt_ax.set_title(title)
        # plt_ax.title("Nterms: "+str(Nterms))
        if phasebinonly:
            pass
        else:
            if RV:
                plt_ax.errorbar(phased_t, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
                plt_ax.errorbar(phased_t + 1, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
            else:
                plt_ax.errorbar(phased_t, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
                plt_ax.errorbar(phased_t + 1, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)

                if plot_rejected:
                    plt_ax.errorbar(phase_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)
                    plt_ax.errorbar(phase_bad + 1, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)
            # plt_ax.errorbar(phased_t+2, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)

            # N_terms
        plt_ax.plot(phase_fit, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')
        plt_ax.plot(phase_fit + 1, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')
        # plt_ax.plot(phase_fit+2, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        plt_ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        if show_err_lines:
            plt_ax.axhline(fmagmn + 3 * ferrmn, color='g', ls='-.', lw=2, alpha=0.5, label='3X Mag Err')
            plt_ax.axhline(fmagmn - 3 * ferrmn, color='g', ls='-.', lw=2, alpha=0.5)
            plt_ax.axhline(fmagmn + 3 * fmag_stdev, color='b', ls=':', lw=2, alpha=0.5, label='3X Mag StDev')
            plt_ax.axhline(fmagmn - 3 * fmag_stdev, color='b', ls=':', lw=2, alpha=0.5)
        else:
            pass
        # plt_ax.set_xlim(0.0, 2.0)
        # plt_ax.set_ylim(18.2, 16.7)
        if RV:
            plt_ax.set_ylabel('RV [km s$^{-1}$]')
        else:
            plt_ax.set_ylabel('mag')
            # plt_ax.set_xlabel('Phase')
            plt_ax.invert_yaxis()

        plt_ax.grid()
        if phasebin:
            plt_ax.errorbar(binned_phase, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=1.5, alpha=0.3)
            plt_ax.errorbar(binned_phase + 1, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=1.5, alpha=0.3)

        if plt_resid:
            frame1.set_xlim(0.0, 2.0)
            frame1.set_xticklabels([])  # Remove x-tic labels for the first frame
            frame2 = fig.add_axes((.1, .1, .8, .2))

            frame2.errorbar(phased_t, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=0.3)
            frame2.errorbar(phased_t + 1, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=0.3)
            # plt_ax.errorbar(phased_t+2, resid, fmt='.k', ecolor='k', lw=1, ms=4, capsize=1.5, alpha=0.3)

            frame2.grid()
            # ax2 = plt.gca()
            frame2.set_xlim(0.0, 2.0)
            frame2.set_ylim(1.5 * resid.min(), 1.5 * resid.max())
            # frame2.yaxis.set_major_locator(plt.MaxNLocator(4))

            frame2.set_xlabel(r'Phase')
            frame2.set_ylabel('Residual')  # \n N$_{terms} = 4$')
        else:
            plt_ax.set_xlabel('Phase')
    else:
        # fig1 = plt.figure()
        plt_ax = plt.gca()
        plt_ax.set_title(title)
        if RV:
            plt_ax.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
        else:
            plt_ax.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
            if plot_rejected:
                plt_ax.errorbar(mjd_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        plt_ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        plt_ax.axhline(fmagmn + 3 * ferrmn, color='g', ls='-.', lw=2, alpha=0.5, label='3X Mag Err')
        plt_ax.axhline(fmagmn - 3 * ferrmn, color='g', ls='-.', lw=2, alpha=0.5)
        plt_ax.axhline(fmagmn + 3 * fmag_stdev, color='b', ls=':', lw=2, alpha=0.5, label='3X Mag StDev')
        plt_ax.axhline(fmagmn - 3 * fmag_stdev, color='b', ls=':', lw=2, alpha=0.5)

        # plt_ax.set_xlim(0.0, 2.0)
        # plt_ax.set_ylim(18.2, 16.7)
        plt_ax.set_xlabel('MJD')
        plt_ax.grid()
        if RV:
            plt_ax.set_ylabel('RV [km s$^{-1}$]')
        else:
            plt_ax.set_ylabel('mag')
            plt_ax.invert_yaxis()


def plt_any_lc_ax(lc_data, P, ax, is_Periodic=False, title="", stdTitle=True, phasebin=False, bins=25, RV=False, phasebinonly=False, show_err_lines=True, plot_rejected=False):
    goodQualIndex = np.where(lc_data['QualFlag']==True)[0]
    badQualIndex = np.where(lc_data['QualFlag']==False)[0]
    mjd = lc_data['mjd'][goodQualIndex].data
    mag = lc_data['mag'][goodQualIndex].data
    err = lc_data['magerr'][goodQualIndex].data

    data = [mjd, mag, err]
    AFD_data = AFD(data, P)

    if len(AFD_data) == 8:
        deasliased_period, Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiS, mfit = AFD_data
    else:
        Nterms, phase_fit, y_fit, phased_t, resid, reduced_ChiS, mfit = AFD_data

    Amp = y_fit.max() - y_fit.min()
    t0 = (mjd - (phased_t * P)).min()

    mean_mag = np.nanmean(mag)
    if stdTitle:
        title = title + "P = {!s}d $|$ $m$ = {!s} $|$ Amp = {!s} $|$ t0 = {!s}".format(np.round(P, 7), np.round(mean_mag, 2), np.round(Amp, 3), np.round(t0, 7))
    else:
        title = title + "P = {!s}d $|$ Amp = {!s} $|$ t0 = {!s}".format(np.round(P, 4), np.round(Amp, 3), np.round(t0, 5))

    mjd_bad = lc_data['mjd'][badQualIndex].data
    phase_bad = ((mjd_bad - t0) / P) % 1
    mag_bad = lc_data['mag'][badQualIndex].data
    err_bad = lc_data['magerr'][badQualIndex].data

    binned_phase, binned_mag, binned_err = bin_phaseLC(phased_t, mag, err, bins=bins)

    plt_ax = ax
    # is_Periodic = True
    if  is_Periodic:
        plt_ax.set_title(title)
        #plt_ax.title("Nterms: "+str(Nterms))
        if phasebinonly:
            pass
        else:
            if RV:
                plt_ax.errorbar(phased_t, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
                plt_ax.errorbar(phased_t+1, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
            else:
                plt_ax.errorbar(phased_t, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
                plt_ax.errorbar(phased_t+1, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)

                if plot_rejected:
                    plt_ax.errorbar(phase_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)
                    plt_ax.errorbar(phase_bad+1, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)
            #plt_ax.errorbar(phased_t+2, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)

            # N_terms
        plt_ax.plot(phase_fit, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')
        plt_ax.plot(phase_fit+1, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')
        #plt_ax.plot(phase_fit+2, y_fit, 'b', markeredgecolor='b', lw=2, fillstyle='top', linestyle='solid')

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        
        plt_ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        if show_err_lines:
            plt_ax.axhline(fmagmn+3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5, label='3X Mag Err')
            plt_ax.axhline(fmagmn-3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5)
            plt_ax.axhline(fmagmn+3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5, label='3X Mag StDev')
            plt_ax.axhline(fmagmn-3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5)
        else:
            pass
        #plt_ax.set_xlim(0.0, 2.0)
        #plt_ax.set_ylim(18.2, 16.7)
        if RV:
            plt_ax.set_ylabel('RV [km s$^{-1}$]')
        else:
            plt_ax.set_ylabel('mag')
            #plt_ax.set_xlabel('Phase')
            plt_ax.invert_yaxis()

        plt_ax.grid()
        if phasebin:
            plt_ax.errorbar(binned_phase, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=1.5, alpha=0.3)
            plt_ax.errorbar(binned_phase+1, binned_mag, binned_err, fmt='sr', ecolor='red', lw=1, ms=4, capsize=1.5, alpha=0.3)


        plt_ax.set_xlabel('Phase')
    else:
        plt_ax.set_title(title)
        if RV:
            plt_ax.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5)
        else:
            plt_ax.errorbar(mjd, mag, err, fmt='.k', ecolor='gray', lw=1, ms=4, capsize=1.5, alpha=0.3)
            if plot_rejected:
                plt_ax.errorbar(mjd_bad, mag_bad, err_bad, fmt='.r', ecolor='r', lw=1, ms=4, capsize=1.5, alpha=0.3)

        fmagmn = np.mean(mag)
        ferrmn = np.mean(err)
        fmag_stdev = np.std(mag)

        plt_ax.axhline(fmagmn, color='r', ls='-', lw=2, label='Mean Mag')
        plt_ax.axhline(fmagmn+3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5, label='3X Mag Err')
        plt_ax.axhline(fmagmn-3*ferrmn, color='g', ls='-.', lw=2 ,alpha=0.5)
        plt_ax.axhline(fmagmn+3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5, label='3X Mag StDev')
        plt_ax.axhline(fmagmn-3*fmag_stdev, color='b', ls=':', lw=2,alpha=0.5)

        #plt_ax.set_xlim(0.0, 2.0)
        #plt_ax.set_ylim(18.2, 16.7)
        plt_ax.set_xlabel('MJD')
        plt_ax.grid()
        if RV:
            plt_ax.set_ylabel('RV [km s$^{-1}$]')
        else:
            plt_ax.set_ylabel('mag')
            plt_ax.invert_yaxis()


def plot_powerspec(frequency, power, ax1=None, ax2=None, FAP_power_peak=None, logFAP_limit=None, alias_df=None, title=""):
    minimum_frequency = frequency.min()
    maximum_frequency = frequency.max()

    minP = maximum_frequency**-1
    maxP = minimum_frequency**-1

    if (ax1 is None) and (ax2 is None):
        fig = plt.figure(figsize=(23, 8), constrained_layout=True)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])

    ax1.set_title(title)
    ax1.plot(frequency.to(u.microHertz), power, c='k', lw=0.75)
    if FAP_power_peak is not None:
        ax1.axhline(y=FAP_power_peak, c='r', ls='dashed', alpha=0.5, lw=0.75)
        if logFAP_limit is not None:
            ax1.text(0.8 * maximum_frequency.to(u.microHertz).value, FAP_power_peak - 0.05, f"log(FAP) = {logFAP_limit}", c='r')

    xmin = minimum_frequency.to(u.microHertz).value
    xmax = maximum_frequency.to(u.microHertz).value

    ax1.axvline(x=((365.25 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((29.530587981 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((27.321661 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 2 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 3 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 4 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 5 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.set_xlabel(r'Frequency [$\mu$Hz]')
    ax1.set_ylabel('Power')
    ax1.set_xlim((xmin, xmax))

    xmin = minimum_frequency.to(u.microHertz).value
    xmax = maximum_frequency.to(u.microHertz).value
    dx = xmax - xmin
    np.ceil(dx)

    xmajortick = np.floor(np.ceil(dx) / 23)
    xminortick = xmajortick / 10

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(xmajortick))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(xminortick))

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    if alias_df is not None:
        n = [-4, -3, -2, -1, 1, 2, 3, 4]
        f0 = frequency[np.argmax(power)]
        ax1.axvline(x=f0.to(u.microHertz).value, c='r', ls='dashed', alpha=0.75, lw=0.75)
        for ii in n:
            ax1.axvline(x=np.abs((f0 + (ii * alias_df)).to(u.microHertz).value), c='b', ls='dashed', alpha=0.5, lw=0.75)
            if ii > 1:
                ax1.axvline(x=(f0 / ii).to(u.microHertz).value, c='g', ls='dashed', alpha=0.5, lw=0.75)
                ax1.axvline(x=(f0 * ii).to(u.microHertz).value, c='g', ls='dashed', alpha=0.5, lw=0.75)

    ax2.plot((1.0 / frequency).to(u.d), power, c='k', lw=0.75)
    if FAP_power_peak is not None:
        ax2.axhline(y=FAP_power_peak, c='r', ls='dashed', alpha=0.5, lw=0.75)
        if logFAP_limit is not None:
            ax2.text(0.1 * maxP.to(u.d).value, FAP_power_peak - 0.05, f"log(FAP) = {logFAP_limit}", c='r')

    ax2.axvline(x=365.25, c='k', ls='dashed', alpha=0.2, lw=0.75)
    ax2.axvline(x=29.530587981, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax2.axvline(x=27.321661, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax2.axvline(x=1, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax2.axvline(x=1 / 2, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax2.axvline(x=1 / 3, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax2.axvline(x=1 / 4, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax2.axvline(x=1 / 5, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax2.set_xlabel('Period [d]')
    ax2.set_ylabel('Power')
    ax2.set_xscale('log')
    ax2.set_xlim((minP.to(u.d).value, maxP.to(u.d).value))

    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    if alias_df is not None:
        n = [-4, -3, -2, -1, 1, 2, 3, 4]
        f0 = frequency[np.argmax(power)]
        ax2.axvline(x=(f0**-1).to(u.d).value, c='r', ls='dashed', alpha=0.75, lw=0.75)
        for ii in n:
            ax2.axvline(x=np.abs((f0 + (ii * alias_df)).to(1 / u.d).value)**-1, c='b', ls='dashed', alpha=0.5, lw=0.75)
            if ii > 1:
                ax2.axvline(x=(f0 / ii).to(1 / u.d).value**-1, c='g', ls='dashed', alpha=0.5, lw=0.75)
                ax2.axvline(x=(f0 * ii).to(1 / u.d).value**-1, c='g', ls='dashed', alpha=0.5, lw=0.75)
    return ax1, ax2



def plot_single_powerspec(frequency, power, ax1=None, FAP_power_peak=None, logFAP_limit=None, alias_df=None, title=""):
    minimum_frequency = frequency.min()
    maximum_frequency = frequency.max()

    minP = maximum_frequency**-1
    maxP = minimum_frequency**-1

    if (ax1 is None):
        fig = plt.figure(figsize=(23, 8), constrained_layout=True)
        gs = GridSpec(1, 1, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])

    ax1.set_title(title)
    ax1.plot(frequency.to(u.microHertz), power, c='k', lw=0.75)
    if FAP_power_peak is not None:
        ax1.axhline(y=FAP_power_peak, c='r', ls='dashed', alpha=0.5, lw=0.75)
        if logFAP_limit is not None:
            ax1.text(0.8 * maximum_frequency.to(u.microHertz).value, FAP_power_peak - 0.05, f"log(FAP) = {logFAP_limit}", c='r')

    xmin = minimum_frequency.to(u.microHertz).value
    xmax = maximum_frequency.to(u.microHertz).value

    ax1.axvline(x=((365.25 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((29.530587981 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((27.321661 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 2 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 3 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 4 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.axvline(x=((1 / 5 * u.d)**-1).to(u.microHertz).value, c='k', ls='dashed', alpha=0.5, lw=0.75)
    ax1.set_xlabel(r'Frequency [$\mu$Hz]')
    ax1.set_ylabel('Power')
    ax1.set_xlim((xmin, xmax))

    xmin = minimum_frequency.to(u.microHertz).value
    xmax = maximum_frequency.to(u.microHertz).value
    dx = xmax - xmin
    np.ceil(dx)

    xmajortick = np.floor(np.ceil(dx) / 23)
    xminortick = xmajortick / 10

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(xmajortick))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(xminortick))

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    if alias_df is not None:
        n = [-4, -3, -2, -1, 1, 2, 3, 4]
        f0 = frequency[np.argmax(power)]
        ax1.axvline(x=f0.to(u.microHertz).value, c='r', ls='dashed', alpha=0.75, lw=0.75)
        for ii in n:
            ax1.axvline(x=np.abs((f0 + (ii * alias_df)).to(u.microHertz).value), c='b', ls='dashed', alpha=0.5, lw=0.75)
            if ii > 1:
                ax1.axvline(x=(f0 / ii).to(u.microHertz).value, c='g', ls='dashed', alpha=0.5, lw=0.75)
                ax1.axvline(x=(f0 * ii).to(u.microHertz).value, c='g', ls='dashed', alpha=0.5, lw=0.75)

    return ax1


def plot_LC_analysis(lc_data, P, frequency, power, FAP_power_peak=None, logFAP_limit=None, df=None, title=""):
    fig = plt.figure(figsize=(12, 9), constrained_layout=False)
    gs = GridSpec(4, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2:, 0])
    ax4 = fig.add_subplot(gs[2:, 1])

    plot_powerspec(frequency, power, ax1=ax1, ax2=ax2, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, alias_df=df, title=title)
    plt_any_lc_ax(lc_data, P, is_Periodic=False, ax=ax3, title="", phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, P, is_Periodic=True, ax=ax4, title="", phasebin=True, bins=25)

    ax3.set_title("")
    plt.tight_layout()


def plot_LC_analysis_aliases(lc_data, P, frequency, power, FAP_power_peak=None, logFAP_limit=None, df=None, title=""):
    fig = plt.figure(figsize=(18, 9), constrained_layout=False)
    gs = GridSpec(4, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[1, :2])

    ax3 = fig.add_subplot(gs[:2, 2])

    ax4 = fig.add_subplot(gs[2:, 0])
    ax5 = fig.add_subplot(gs[2:, 1])
    ax6 = fig.add_subplot(gs[2:, 2])

    plot_powerspec(frequency, power, ax1=ax1, ax2=ax2, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, alias_df=df, title=title)

    plt_any_lc_ax(lc_data, P, is_Periodic=False, ax=ax3, title="", phasebin=True, bins=25)

    plt_any_lc_ax(lc_data, 0.5*P, is_Periodic=True, ax=ax4, title="", phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 1.0*P, is_Periodic=True, ax=ax5, title="", phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 2.0* P, is_Periodic=True, ax=ax6, title="", phasebin=True, bins=25)

    ax3.set_title("")
    plt.tight_layout()


def plot_LC_analysis_ALLaliases(lc_data, P, frequency, power, FAP_power_peak=None, logFAP_limit=None, df=None, title=""):
    fig = plt.figure(figsize=(24, 12), constrained_layout=False)
    gs = GridSpec(6, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[1, :2])

    ax3 = fig.add_subplot(gs[:2, 2])

    ax4 = fig.add_subplot(gs[:2, 3])

    ax5 = fig.add_subplot(gs[2:4, 0])
    ax6 = fig.add_subplot(gs[2:4, 1])
    ax7 = fig.add_subplot(gs[2:4, 2])
    ax8 = fig.add_subplot(gs[2:4, 3])

    ax9 = fig.add_subplot(gs[4:, 0])
    ax10 = fig.add_subplot(gs[4:, 1])
    ax11 = fig.add_subplot(gs[4:, 2])
    ax12 = fig.add_subplot(gs[4:, 3])

    plot_powerspec(frequency, power, ax1=ax1, ax2=ax2, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, alias_df=df, title=title)

    plt_any_lc_ax(lc_data, P, is_Periodic=False, ax=ax3, title="", phasebin=True, bins=25)

    plt_any_lc_ax(lc_data, 1.0*P, is_Periodic=True, ax=ax4, title="", phasebin=True, bins=25)

    plt_any_lc_ax(lc_data, 0.5*P, is_Periodic=True, ax=ax5, title="(1/2)", phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 2.0* P, is_Periodic=True, ax=ax9, title="2", phasebin=True, bins=25)

    plt_any_lc_ax(lc_data, (1/3)*P, is_Periodic=True, ax=ax6, title="(1/3)", phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 3.0* P, is_Periodic=True, ax=ax10, title="3", phasebin=True, bins=25)

    plt_any_lc_ax(lc_data, 0.25*P, is_Periodic=True, ax=ax7, title="(1/4)", phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 4.0* P, is_Periodic=True, ax=ax11, title="4", phasebin=True, bins=25)

    plt_any_lc_ax(lc_data, 0.2*P, is_Periodic=True, ax=ax8, title="(1/5)", phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 5.0* P, is_Periodic=True, ax=ax12, title="5", phasebin=True, bins=25)

    ax3.set_title("")
    plt.tight_layout()


def plot_LC_analysis_Pal2013(lc_data, P1, P2, P3, frequency, power, FAP_power_peak=None, logFAP_limit=None, df=None, title=""):
    fig = plt.figure(figsize=(32, 18), constrained_layout=False)
    gs = GridSpec(8, 5, figure=fig)

    ax1 = fig.add_subplot(gs[0, :4])
    ax2 = fig.add_subplot(gs[1, :4])

    ax3 = fig.add_subplot(gs[:2, 4])

    ax4 = fig.add_subplot(gs[2:4, 0])
    ax5 = fig.add_subplot(gs[2:4, 1])
    ax6 = fig.add_subplot(gs[2:4, 2])
    ax7 = fig.add_subplot(gs[2:4, 3])
    ax8 = fig.add_subplot(gs[2:4, 4])

    ax9 = fig.add_subplot(gs[4:6, 0])
    ax10 = fig.add_subplot(gs[4:6, 1])
    ax11 = fig.add_subplot(gs[4:6, 2])
    ax12 = fig.add_subplot(gs[4:6, 3])
    ax13 = fig.add_subplot(gs[4:6, 4])

    ax14 = fig.add_subplot(gs[6:8, 0])
    ax15 = fig.add_subplot(gs[6:8, 1])
    ax16 = fig.add_subplot(gs[6:8, 2])
    ax17 = fig.add_subplot(gs[6:8, 3])
    ax18 = fig.add_subplot(gs[6:8, 4])

    plot_powerspec(frequency, power, ax1=ax1, ax2=ax2, FAP_power_peak=FAP_power_peak, logFAP_limit=logFAP_limit, alias_df=df, title=title)

    plt_any_lc_ax(lc_data, P1, is_Periodic=False, ax=ax3, title="", phasebin=True, bins=25)

    plt_any_lc_ax(lc_data, (1 / 3) * P1, is_Periodic=True, ax=ax4, title="f = {!s}$\mu$Hz $|$ 0.33".format(np.round(((P1 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, (1 / 2) * P1, is_Periodic=True, ax=ax5, title="f = {!s}$\mu$Hz $|$ 0.5".format(np.round(((P1 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 1.0 * P1, is_Periodic=True, ax=ax6, title="f = {!s}$\mu$Hz $|$ ".format(np.round(((P1 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 2.0 * P1, is_Periodic=True, ax=ax7, title="f = {!s}$\mu$Hz $|$ 2".format(np.round(((P1 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 3.0 * P1, is_Periodic=True, ax=ax8, title="f = {!s}$\mu$Hz $|$ 3".format(np.round(((P1 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)

    plt_any_lc_ax(lc_data, (1 / 3) * P2, is_Periodic=True, ax=ax9, title="f = {!s}$\mu$Hz $|$ 0.33".format(np.round(((P2 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, (1 / 2) * P2, is_Periodic=True, ax=ax10, title="f = {!s}$\mu$Hz $|$ 0.5".format(np.round(((P2 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 1.0 * P2, is_Periodic=True, ax=ax11, title="f = {!s}$\mu$Hz $|$ ".format(np.round(((P2 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 2.0 * P2, is_Periodic=True, ax=ax12, title="f = {!s}$\mu$Hz $|$ 2".format(np.round(((P2 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 3.0 * P2, is_Periodic=True, ax=ax13, title="f = {!s}$\mu$Hz $|$ 3".format(np.round(((P2 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)

    plt_any_lc_ax(lc_data, (1 / 3) * P3, is_Periodic=True, ax=ax14, title="f = {!s}$\mu$Hz $|$ 0.33".format(np.round(((P3 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, (1 / 2) * P3, is_Periodic=True, ax=ax15, title="f = {!s}$\mu$Hz $|$ 0.5".format(np.round(((P3 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 1.0 * P3, is_Periodic=True, ax=ax16, title="f = {!s}$\mu$Hz $|$ ".format(np.round(((P3 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 2.0 * P3, is_Periodic=True, ax=ax17, title="f = {!s}$\mu$Hz $|$ 2".format(np.round(((P3 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)
    plt_any_lc_ax(lc_data, 3.0 * P3, is_Periodic=True, ax=ax18, title="f = {!s}$\mu$Hz $|$ 3".format(np.round(((P3 * u.d)**-1).to(u.microHertz),2)), stdTitle=False, phasebin=True, bins=25)

    ax3.set_title("")
    plt.tight_layout()




if (__name__ == '__main__'):
    print('LCtools.py is not meant to be run. Please import only.')
