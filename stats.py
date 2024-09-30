# Файл содержит вспомогательные статистические функции

import numpy
import scipy
from matplotlib import mlab
import matplotlib.pyplot as plt


def bandpass(trials:numpy.ndarray, low:float, high:float, sample_rate:float):
    '''
    Designs and applies a bandpass filter to the signal.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal
    lo : float
        Lower frequency bound (in Hz)
    hi : float
        Upper frequency bound (in Hz)
    sample_rate : float
        Sample rate of the signal (in Hz)
    
    Returns
    -------
    trials_filt : 3d-array (channels x samples x trials)
        The bandpassed signal
    '''

    # The iirfilter() function takes the filter order: higher numbers mean a sharper frequency cutoff,
    # but the resulting signal might be shifted in time, lower numbers mean a soft frequency cutoff,
    # but the resulting signal less distorted in time. It also takes the lower and upper frequency bounds
    # to pass, divided by the niquist frequency, which is the sample rate divided by 2:
    a, b = scipy.signal.iirfilter(6, [low/(sample_rate/2.0), high/(sample_rate/2.0)])

    # Applying the filter to each trial
    n_channels:int = trials.shape[0]
    n_samples:int = trials.shape[1]
    n_trials:int = trials.shape[2]
    trials_filt = numpy.zeros((n_channels, n_samples, n_trials))
    for i in range(n_trials):
        trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)
    
    return trials_filt

def cov(trials):
    ''' Calculate the covariance for each trial and return their average '''
    n_trials = trials.shape[2]
    covs = [ trials[:,:,i].dot(trials[:,:,i].T) / trials.shape[1] for i in range(n_trials) ]
    return numpy.mean(covs, axis=0)

def whitening(sigma):
    ''' Calculate a whitening matrix for covariance matrix sigma. '''
    U, l, _ = numpy.linalg.svd(sigma)
    return U.dot(numpy.diag(l ** -0.5) )

def csp(trials_l, trials_r):
    '''
    Calculate the CSP transformation matrix W.
    arguments:
        trials_l - Array (channels x samples x trials) containing left ear trials
        trials_r - Array (channels x samples x trials) containing right ear trials
    returns:
        Mixing matrix W
    '''
    cov_l = cov(trials_l)
    cov_r = cov(trials_r)
    P = whitening(cov_l + cov_r)
    B, _, _ = numpy.linalg.svd(P.T.dot(cov_r).dot(P))
    W = P.dot(B)
    return W

def apply_mix(W, trials):
    ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
    n_channels = trials.shape[0]
    n_samples = trials.shape[1]
    n_trials = trials.shape[2]
    trials_csp = numpy.zeros((n_channels, n_samples, n_trials))
    for i in range(n_trials):
        trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
    return trials_csp

def psd(trials, sample_rate:float):
    '''
    Calculates for each trial the Power Spectral Density (PSD).
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal
    
    Returns
    -------
    trial_PSD : 3d-array (channels x PSD x trials)
        the PSD for each trial.  
    freqs : list of floats
        The frequencies for which the PSD was computed (useful for plotting later)
    '''
    
    n_channels = trials.shape[0]
    n_samples = trials.shape[1]
    n_trials = trials.shape[2]
    trials_PSD = numpy.zeros((n_channels, 51, n_trials))

    # Iterate over trials and channels
    for trial in range(n_trials):
        for ch in range(n_channels):
            # Calculate the PSD
            (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(n_samples), Fs=sample_rate)
            trials_PSD[ch, :, trial] = PSD.ravel()
                
    return trials_PSD, freqs

def plot_psd(trials_PSD, trials_filt, freqs, chan_ind, chan_lab=None, maxy=None):
    '''
    Plots PSD data calculated with psd().
    
    Parameters
    ----------
    trials : 3d-array
        The PSD data, as returned by psd()
    freqs : list of floats
        The frequencies for which the PSD is defined, as returned by psd() 
    chan_ind : list of integers
        The indices of the channels to plot
    chan_lab : list of strings
        (optional) List of names for each channel
    maxy : float
        (optional) Limit the y-axis to this value
    '''
    plt.figure(figsize=(12,5))
    
    nchans = len(chan_ind)
    
    # Maximum of 3 plots per row
    nrows = int(numpy.ceil(nchans / 3))
    ncols = min(3, nchans)
    
    # Enumerate over the channels
    for i,ch in enumerate(chan_ind):
        # Figure out which subplot to draw to
        plt.subplot(nrows,ncols,i+1)
    
        # Plot the PSD for each class
        for cl in trials_filt.keys():
            plt.plot(freqs, numpy.mean(trials_PSD[cl][ch,:,:], axis=1), label=cl)
    
        # All plot decoration below...
        
        plt.xlim(1,30)
        
        if maxy != None:
            plt.ylim(0,maxy)
    
        plt.grid()
    
        plt.xlabel('Frequency (Hz)')
        
        if chan_lab == None:
            plt.title('Channel %d' % (ch+1))
        else:
            plt.title(chan_lab[i])

        plt.legend()
        
    plt.tight_layout()
    plt.savefig('./results/psd_plot.png')

def plot_logvar(trials, trials_filt):
    '''
    Plots the log-var of each channel/component.
    arguments:
        trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
    '''
    plt.figure(figsize=(12,5))
    nchannels = trials_filt['left'].shape[0]

    x0 = numpy.arange(nchannels)
    x1 = numpy.arange(nchannels) + 0.4

    y0 = numpy.mean(trials['left'], axis=1)
    y1 = numpy.mean(trials['right'], axis=1)

    plt.bar(x0, y0, width=0.5, color='b')
    plt.bar(x1, y1, width=0.4, color='r')

    plt.xlim(-0.5, nchannels+0.5)

    plt.gca().yaxis.grid(True)
    plt.title('log-var of each channel/component')
    plt.xlabel('channels/components')
    plt.ylabel('log-var')
    plt.legend(['left', 'rigth'])
    plt.savefig('./results/logvar.png')