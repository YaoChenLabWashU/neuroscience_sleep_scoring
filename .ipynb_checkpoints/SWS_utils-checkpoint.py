import os
import numpy as np
import sys
from scipy.integrate import simpson as simps
from scipy import signal, io
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.image as mpimg
import matplotlib.patches as patch
import copy as cp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import joblib
import pandas as pd
pd.options.mode.chained_assignment = None
import cv2
import math
from pylab import *
from matplotlib import *
import time
import glob
from dateutil.parser import parse
from datetime import datetime, timedelta
import json
import seaborn as sns
import shutil
import matplotlib.colors as mcolors
import PKA_Sleep as PKA

def generate_signal(downsamp_signal, epochlen, fs): # fs = fsd here
    # mean of 4 seconds
    normmean = np.mean(downsamp_signal)
    normstd = np.std(downsamp_signal)
    binl = epochlen * fs  # bin size in array slots | number of points in an epoch; a bin == an epoch
    sig_var = np.zeros(int(np.size(downsamp_signal) / (epochlen * fs)))
    sig_mean = np.zeros(int(np.size(downsamp_signal) / (epochlen  * fs)))
    sig_max = np.zeros(int(np.size(downsamp_signal) / (epochlen  * fs)))

    for i in np.arange(np.size(sig_var)):
        sig_var[i] = np.var(downsamp_signal[epochlen  * fs * (i):(epochlen  * fs * (i + 1))])
        sig_mean[i] = np.mean(np.abs(downsamp_signal[epochlen  * fs * (i):(epochlen  * fs * (i + 1))]))
        sig_max[i] = np.max(downsamp_signal[epochlen  * fs * (i):(epochlen  * fs * (i + 1))])

    sig_var = (sig_var - normmean) / normstd # normalization
    sig_max = (sig_max - np.average(sig_max)) / np.std(sig_max)

    # we do not normalize mean (for some reason)
    
    return sig_var, sig_max, sig_mean

def bandPower(this_eeg, fsd, freq_dict = None, minfreq = 0.5, maxfreq = 16, window_length = 10, 
    noverlap = 9.9, window_type = None):
    power_dict = {}
    Pxx, freqs, bins = plot_spectrogram(None, this_eeg, fsd, minfreq = minfreq, maxfreq = maxfreq, 
        window_length = window_length, noverlap = noverlap, window_type = window_type)
    freq_res = freqs[1]-freqs[0]
    if freq_dict:
        for k in list(freq_dict.keys()):
            print('Calculating ' + k + ' Band Power...')
            idx_low = freq_dict[k][0]
            idx_high = freq_dict[k][1]
            power_dict[k] = simps(Pxx[np.where(np.logical_and(freqs>=idx_low,freqs<=idx_high))], 
                axis = 0, dx=freq_res)
    power_dict['Total_Power'] = simps(Pxx, dx=freq_res, axis = 0)
    power_dict['Bins'] = bins
    
    return power_dict

def peak_freq(this_eeg, fsd, freq_dict = None, minfreq = 0.5, maxfreq = 16, window_length = 10, 
    noverlap = 9.9, window_type = None):

    Pxx, freqs, bins = plot_spectrogram(None, this_eeg, fsd, minfreq = minfreq, maxfreq = maxfreq, 
        window_length = window_length, noverlap = noverlap, window_type = window_type)
    peak_freqs_overall = np.zeros(np.shape(Pxx)[1])
    # peak_theta = np.zeros(np.shape(Pxx)[1])
    for f in np.arange(0, np.shape(Pxx)[1]):
        # theta_band = Pxx[np.where(np.logical_and(freqs>=5,freqs<=8))]
        # theta_freqs = freqs[np.where(np.logical_and(freqs>=5,freqs<=8))]
        peak_idx_overall = np.argmax(Pxx[:,f])
        # peak_idx_theta = np.argmax(theta_band[:,f])
        peak_freqs_overall[f] = freqs[peak_idx_overall]
        # peak_theta[f] = theta_freqs[peak_idx_theta]
    return peak_idx_overall

def post_pre(post, pre):
	post = np.append(post, 0)
	post = np.delete(post, 0)
	pre = np.append(0, pre)
	pre = pre[0:-1]
	return post, pre

### functions used in the Hengen code

def fix_states(states, alter_nums = False):
	if alter_nums == True:
		states[states == 1] = 0
		states[states == 3] = 5

	for ss in np.arange(np.size(states)-1):
		#check if it is a flicker state
		if (ss != 0 and ss < np.size(states)-1):
			if states[ss+1] == states[ss-1]:
				states[ss] = states[ss+1]

		if (states[ss] == 0 and states[ss+1] == 5):
			states[ss] = 2
	if alter_nums == True:
		states[states == 0] = 1
		states[states == 5] = 3
	return states

def random_forest_classifier(features, target):
    clf = RandomForestClassifier(n_estimators=300)
    # clf.fit(preprocessing.LabelEncoder().fit_transform(features), target)
    clf.fit(features, target)
    return clf

def plot_spectrogram(ax, eegdat, fsd, minfreq = 1, maxfreq = 16, additional_ax = None, window_length = 10, 
    noverlap = 9.9, vmin = -50, vmax = -10, window_type = None):
    dt = 1/fsd
    t_elapsed = eegdat.shape[0]/fsd
    t = np.arange(0.0, t_elapsed, dt)
    noverlap = noverlap * fsd
    NFFT = window_length * fsd
    if window_type:
        window_array = window_type(int(NFFT))
    else:
        window_array = None
    if additional_ax:
        additional_ax.set_xlabel('Time (seconds)')
        additional_ax.set_ylabel('Frequency (Hz)')      

    if ax:
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')
    # the minfreq and maxfreq args will limit the frequencies
        Pxx, freqs, bins, im = my_specgram(eegdat, ax = ax, NFFT=int(NFFT), Fs=fsd, noverlap=int(noverlap),
                                    cmap=cm.get_cmap('plasma'), minfreq = minfreq, maxfreq = maxfreq,
                                    xextent = (0,int(t_elapsed)), additional_ax = additional_ax, vmin = vmin, 
                                    vmax = vmax, window = window_array)
        return Pxx, freqs, bins, im
    else:
        Pxx, freqs, bins = my_specgram(eegdat, ax = ax, NFFT=int(NFFT), Fs=fsd, noverlap=int(noverlap),
                                    cmap=cm.get_cmap('plasma'), minfreq = minfreq, maxfreq = maxfreq,
                                    xextent = (0,int(t_elapsed)), additional_ax = additional_ax, vmin = vmin, 
                                    vmax = vmax, window = window_array)
        return Pxx, freqs, bins

def my_specgram(x, ax = None, NFFT=400, Fs=200, Fc=0, detrend=mlab.detrend_none,
             window=None, noverlap=200,
             cmap=None, xextent=None, pad_to=None, sides='default',
             scale_by_freq=None, minfreq = None, maxfreq = None, additional_ax = None, vmin = -50, vmax = -10, 
             **kwargs):
    """
    call signature::

      specgram(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
               window=mlab.window_hanning, noverlap=128,
               cmap=None, xextent=None, pad_to=None, sides='default',
               scale_by_freq=None, minfreq = None, maxfreq = None, **kwargs)

    Compute a spectrogram of data in *x*.  Data are split into
    *NFFT* length segments and the PSD of each section is
    computed.  The windowing function *window* is applied to each
    segment, and the amount of overlap of each segment is
    specified with *noverlap*.

    %(PSD)s

      *Fc*: integer
        The center frequency of *x* (defaults to 0), which offsets
        the y extents of the plot to reflect the frequency range used
        when a signal is acquired and then filtered and downsampled to
        baseband.

      *cmap*:
        A :class:`matplotlib.cm.Colormap` instance; if *None* use
        default determined by rc

      *xextent*:
        The image extent along the x-axis. xextent = (xmin,xmax)
        The default is (0,max(bins)), where bins is the return
        value from :func:`mlab.specgram`

      *minfreq, maxfreq*
        Limits y-axis. Both required

      *kwargs*:

        Additional kwargs are passed on to imshow which makes the
        specgram image

      Return value is (*Pxx*, *freqs*, *bins*, *im*):

      - *bins* are the time points the spectrogram is calculated over
      - *freqs* is an array of frequencies
      - *Pxx* is a len(times) x len(freqs) array of power
      - *im* is a :class:`matplotlib.image.AxesImage` instance

    Note: If *x* is real (i.e. non-complex), only the positive
    spectrum is shown.  If *x* is complex, both positive and
    negative parts of the spectrum are shown.  This can be
    overridden using the *sides* keyword argument.

    **Example:**

    .. plot:: mpl_examples/pylab_examples/specgram_demo.py

    """

    #####################################
    # modified  axes.specgram() to limit
    # the frequencies plotted
    #####################################

    # this will fail if there isn't a current axis in the global scope
    #ax = gca()
    Pxx, freqs, bins = mlab.specgram(x, NFFT, Fs, detrend,
         window, noverlap, pad_to, sides, scale_by_freq)

    # modified here
    #####################################
    if minfreq is not None and maxfreq is not None:
        Pxx = Pxx[(freqs >= minfreq) & (freqs <= maxfreq)]
        freqs = freqs[(freqs >= minfreq) & (freqs <= maxfreq)]
    #####################################

    Z = 10. * np.log10(Pxx)
    Z = np.flipud(Z)

    if xextent is None: xextent = 0, np.amax(bins)
    xmin, xmax = xextent
    freqs += Fc
    extent = xmin, xmax, freqs[0], freqs[-1]
    if not vmin and not vmax:
        vmin = np.percentile(np.concatenate(Z), 2)
        vmax = np.percentile(np.concatenate(Z), 98)
    if ax:
        im = ax.imshow(Z, cmap, extent=extent, **kwargs, vmin = vmin, vmax = vmax)
        print('vmin: '+str(vmin)+'; vmax: '+str(vmax))
        ax.axis('auto')
        xticks = np.arange(100,900,100)*4
        ax.set_xticks(xticks)
        if additional_ax:
            im = additional_ax.imshow(Z, cmap, extent=extent, **kwargs, vmin = vmin, vmax = vmax)
            additional_ax.axis('auto')
            additional_ax.set_xticks(xticks)
        return Pxx, freqs, bins, im
    else:
        return Pxx, freqs, bins

def plot_predicted(ax, Predict_y, is_predicted, clf, Features):
    ax.set_title('Predicted States', x=0.5, y=0.7)
    for state in np.arange(np.size(Predict_y)):
        if Predict_y[state] == 0:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = 'grey')
            ax.add_patch(rect7)           
        if Predict_y[state] == 1:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = 'green')
            ax.add_patch(rect7)
        elif Predict_y[state] == 2:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = 'blue')
            ax.add_patch(rect7)
        elif Predict_y[state] == 3:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = 'red')
            ax.add_patch(rect7)
        elif Predict_y[state] == 4:
            rect7 = patch.Rectangle((state, 0), 3.8, height=1, color='purple')
            ax.add_patch(rect7)
        else:
            print("Model predicted an unknown state.")
    ax.set_ylim(0.3, 1)
    ax.set_yticklabels([])
    ax.set_xlim(0, np.size(Predict_y))
    ax.set_xticks(np.arange(100, np.size(Predict_y), 100))
    ax.tick_params(axis="x",direction="in", pad=-15)
    if is_predicted:
        predictions = clf.predict_proba(Features)
        confidence = np.max(predictions, 1)
    else:
        confidence = np.ones(np.size(Predict_y))
    ax.plot(confidence, color = 'k')

# This is the plotting collection function for the coarse prediction figure
def create_prediction_figure(d, Predict_y, is_predicted, clf, Features, fs, eeg_AD0, eeg_AD2, 
    this_emg, EEG_t, epochlen, start, end, maxfreq, minfreq, additional_axes, v = None):
    plt.ion()
    vmin = d['vmin']
    vmax = d['vmax']

    if vmin == 'None':
        vmin = None

    if vmax =='None':
        vmax = None
    fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows = 5, ncols = 1, figsize = (11, 6))
    if v is not None:
        ax4.plot(v[1], v[0], color = 'k', linestyle = '--')
        ax4.set_ylabel('Velocity')
        ax4.set_ylim([0,40])
        ax4.set_xlim([0,v[1][-1]])

    else:
        ax4.text(0.5, 0.5, 'No Movement Available', 
            horizontalalignment='center', verticalalignment='center')
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    Pxx, freqs, bins, im = plot_spectrogram(ax1, eeg_AD0, fs, maxfreq = maxfreq, minfreq = minfreq, 
        additional_ax = additional_axes[0], vmin = vmin, vmax = vmax)
    Pxx, freqs, bins, im = plot_spectrogram(ax3, eeg_AD2, fs, maxfreq = maxfreq, minfreq = minfreq, 
        additional_ax = additional_axes[1], vmin = vmin, vmax = vmax)

    ax1.xaxis.set_ticks_position('top')
    ax3.set_xticklabels([])

    plot_predicted(ax2, Predict_y, is_predicted, clf, Features)

    ax5.plot(EEG_t, this_emg, color= 'r')
    ax5.set_xlim([EEG_t[0],EEG_t[-1]])
    ax5.set_ylabel('EMG Amplitude')
    fig1.tight_layout()
    fig1.subplots_adjust(wspace=0, hspace=0)
    
    return fig1, ax1, ax2, ax3, ax4, ax5

def update_sleep_df(model_dir, mod_name, df_additions):
    try:
        Sleep_Model = np.load(file = model_dir + mod_name + '_model.pkl', allow_pickle = True)
        Sleep_Model = pd.concat([Sleep_Model,df_additions], ignore_index = True)
    except FileNotFoundError:
        print('no model created...I will save this one')
        Sleep_Model = df_additions
    Sleep_Model.to_pickle(model_dir + mod_name + '_model.pkl')
    return Sleep_Model

def build_joblib_name(d):
    if d['emg']:
        jobname = d['mod_name'] + '_EMG'
        print("EMG flag on")
    else:
        jobname = d['mod_name'] + '_noEMG'
        print('Just so you know...this model has no EMG')
    if d['movement']:
        jobname = jobname + '_movement'
    else:
        jobname = jobname + '_nomovement'
    if len(d['EEG channel']) == 2:
        jobname = jobname + '_2chan'
    jobname = jobname + '.joblib'

    return jobname

def get_xfeatures(FeatureDict):
    x_features = list(FeatureDict.keys())
    try:
        x_features.remove('State')
    except ValueError:
        pass
    try:   
        x_features.remove('animal_name')
    except ValueError:
        pass
    return x_features

def retrain_model(Sleep_Model, x_features, model_dir, jobname):
    print("Retrain model")
    prop = 1 / 2
    model_inputs = Sleep_Model[x_features][0:int((max(Sleep_Model.index) + 1) * prop)].apply(pd.to_numeric)
    train_x = model_inputs.values
    model_input_states = Sleep_Model['State'][0:int((max(Sleep_Model.index) + 1) * prop)].apply(pd.to_numeric)
    train_y = model_input_states.values

    model_test = Sleep_Model[x_features][int((max(Sleep_Model.index) + 1) * prop):].apply(pd.to_numeric)
    test_x = model_test.values
    model_test_states = Sleep_Model['State'][int((max(Sleep_Model.index) + 1) * prop):].apply(pd.to_numeric)
    test_y = model_test_states.values

    print('Calculating tree...')
    clf = random_forest_classifier(train_x, train_y)
    print("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))
    print("Test Accuracy :: ", accuracy_score(test_y, clf.predict(test_x)))


    clf = random_forest_classifier(Sleep_Model[x_features].apply(pd.to_numeric).values,
                                   Sleep_Model['State'].apply(pd.to_numeric).values)
    print("Train Accuracy :: ", accuracy_score(Sleep_Model['State'].apply(pd.to_numeric).values,
                                               clf.predict(Sleep_Model[x_features].apply(pd.to_numeric).values)))
    joblib.dump(clf, model_dir + jobname)


def pull_up_movie(d, cap, start, end, vid_file, epochlen, this_timestamp):
    v = get_videofn_from_csv(d, this_timestamp['Filename'][start])
    print('Pulling up video: '+v)
    print('starting on frame '+str(start))
    start_sec = this_timestamp['Offset_Time'][start]
    print('starting at second '+str(start_sec))
    print('starting '+ v + ' at ' + str(start_sec) + ' seconds')
    if not cap[v].isOpened():
        print("Error opening video stream or file")
    score_win_sec = [start_sec + epochlen, start_sec + epochlen*2]
    try:
        score_win_idx1 = int(this_timestamp.index[this_timestamp['Offset_Time']>(score_win_sec[0])][0])
        score_win_idx2 = int(this_timestamp.index[this_timestamp['Offset_Time']>(score_win_sec[1])][0])
    except IndexError:
        print('No video availabile during this time.')
        return
    score_win = np.arange(score_win_idx1, int(score_win_idx2))
    for f in np.arange(start, end+200):
        cap[v].set(1, f)
        ret, frame = cap[v].read()
        if ret:
            if f in score_win:
                cv2.putText(frame, "SCORE WINDOW", (50, 105), cv2.FONT_HERSHEY_PLAIN, 4, (225, 0, 0), 2)
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('v'):
                break
    cv2.destroyAllWindows()

    #Creates line objects for the fine graph that plots data over 12s intervals
def create_zoomed_fig(ax8, ax9, ax10, long_emg, long_emg_t, long_ThD, long_ThD_t, long_v, long_v_t, 
    start_trace, end_trace, epochlen, ThD_ylims = None, emg_ylims = None, v_ylims = None):

    
    #Delta-Theta
    line1 = plot_zoomed_data(ax8, long_ThD, long_ThD_t, start_trace, end_trace, color = '#5170d7', 
        epochlen = epochlen, ylabel = 'Theta/Delta\nRatio', ylims = ThD_ylims)

    if long_v is None:
        line3 = ax9.plot([0,0], [1,1], linewidth = 0, color = 'w')
        ax9.text(0.5, 0.5, 'No Movement Available', horizontalalignment='center', 
            verticalalignment='center')
    else:
        line3 = plot_zoomed_data(ax9, long_v, long_v_t, start_trace, end_trace, color = '#a87dc2',
            epochlen = epochlen, ylabel = 'Velocity', ylims = v_ylims)

    if long_emg is None:
        ax10.text(0.5, 0.5, 'There is no EMG', horizontalalignment='center', 
            verticalalignment='center')
        line2 = ax10.plot([0,0], [1,1], linewidth = 0, color = 'w')
    else:
        line2 = plot_zoomed_data(ax10, long_emg, long_emg_t, start_trace, end_trace, color = '#fd411e',
            epochlen = epochlen, ylabel = 'EMG Amplitude', linewidth = 2, ylims = emg_ylims)
    plt.show()

    return line1, line2, line3

def plot_zoomed_data(ax, data, t, start_trace, end_trace, color, epochlen, ylabel = None, 
    ylims = None, linewidth = 3):
    start_idx = np.where(t>=start_trace)[0][0]
    end_idx = np.where(t<=end_trace)[0][-1]
    line, = ax.plot(t[start_idx:end_idx+1], data[start_idx:end_idx+1], 
        color = color, linewidth = linewidth)
    ax.set_xlim(t[start_idx], t[end_idx])
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylims:
        ax.set_ylim(ylims)
    else:
        ylims = list(ax.get_ylim())
    h = ylims[-1]-ylims[0]
    rect = patch.Rectangle((0, ylims[0]),epochlen, h, color='#fac205', alpha = 0.3)
    ax.add_patch(rect)    
    return line

def add_buffer(data_array, t_array, buffer_seconds, fs):
    if data_array is None:
        buffered_data = None
        buffered_t = None
    else:
        buffered_data = np.concatenate((np.full(int(buffer_seconds*fs), 0),data_array, np.full(int(buffer_seconds*fs), 0)))
        pre_buffer = np.arange(-buffer_seconds, 0, 1/fs)
        post_buffer = np.arange(t_array[-1]+1/fs, t_array[-1]+buffer_seconds+1/fs, 1/fs)
        buffered_t = np.concatenate([pre_buffer, t_array, post_buffer])
        assert np.size(buffered_data) == np.size(buffered_t)
    return buffered_data, buffered_t
def clear_bins(bins, ax2):
    start_bin = bins[0]
    end_bin = bins[1]
    if end_bin-start_bin == 1:
        end_bin = end_bin+1
    for b in np.arange(start_bin, end_bin-1):
        b = math.floor(b)
        location = b
        rectangle = patch.Rectangle((location, 0), 1.5, height = 2, color = 'white')
        ax2.add_patch(rectangle)
def correct_bins(start_bin, end_bin, ax2, new_state):
    if end_bin-start_bin == 1:
        end_bin = end_bin+1
    for b in np.arange(start_bin, end_bin-1):
        b = math.floor(b)
        location = b
        color = 'white'
        if new_state == 1:
            color = 'green'
        if new_state == 2:
            color = 'blue'
        if new_state == 3:
            color = 'red'
        if new_state == 4:
            color = 'purple'
        print("color: " + str(color))
        rectangle = patch.Rectangle((location, 0), 1.5, height = 2, color = color)
        print('loc: ', location)
        ax2.add_patch(rectangle)

def create_scoring_figure(extracted_dir, a, this_eeg, this_emg, EEG_t, fsd, 
    maxfreq, minfreq, epochlen, v = None, additional_ax = None):
    plt.ion()
    if v is not None:
        fig, (ax1, ax2, ax3, axx, button) = plt.subplots(nrows = 5, ncols = 1, figsize = (16, 8))
        ax3.plot(v[1], v[0], color = 'k', linestyle = '--')
        ax3.set_ylabel('Velocity')
        ax3.set_ylim([0,40])
        ax3.set_xlim([0,int(np.size(EEG_t)/fsd)])
        ax3.set_yticklabels([])
        ax3.set_xticklabels([])
    else:
        fig, (ax1, ax2, axx, button) = plt.subplots(nrows = 4, ncols = 1, figsize = (11, 6))
    Pxx, freqs, bins, im = plot_spectrogram(ax1, this_eeg, fsd, maxfreq = maxfreq, minfreq = minfreq, 
        additional_ax = additional_ax)
    ax1.xaxis.set_ticks_position('top')
    ax2.set_xlim([0,math.ceil(EEG_t[-1]/epochlen)])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    axx.plot(EEG_t, this_emg, color= 'r')
    axx.set_xlim([EEG_t[0],EEG_t[-1]])
    axx.set_ylabel('EMG Amplitude')
    button.set_xlim([0,1])
    button.set_ylim([0,1])

    rect = patch.Rectangle((0,0),1,1, color = 'k')
    button.add_patch(rect)
    button.text(0.5, 0.5, 'Click here for video', horizontalalignment='center', 
        verticalalignment='center', transform=button.transAxes, color = 'w', fontsize = 16)
    button.tick_params(axis='both',which='both',bottom=False,top=False,left = False, 
        labelbottom=False, labelleft = False)

    fig.show()
    if v is not None:
        return fig, ax1, ax2, ax3, axx, button
    else:
        return fig, ax1, ax2, axx, button

def update_raw_trace(fig1, fig2, line1, line2, line3, line4, line5, long_emg, long_emg_t, long_ThD,
 long_ThD_t, long_v, long_v_t, markers, this_epoch_t, start_trace, end_trace, epochlen):

    start_idx_ThD = np.where(long_ThD_t>=start_trace)[0][0]
    end_idx_ThD = np.where(long_ThD_t<=end_trace)[0][-1]

    try:
        assert np.size(line1.get_ydata()) == (end_idx_ThD-start_idx_ThD)+1
    except AssertionError:
        diff = ((end_idx_ThD-start_idx_ThD)+1)-np.size(line1.get_ydata())
        end_idx_ThD = end_idx_ThD-diff

    line1.set_ydata(long_ThD[start_idx_ThD:end_idx_ThD+1])

    if long_emg is not None: 
        start_idx_emg = np.where(long_emg_t>=start_trace)[0][0]
        end_idx_emg = np.where(long_emg_t<=end_trace)[0][-1]
        try:
            assert np.size(line2.get_ydata()) == (end_idx_emg-start_idx_emg)+1
        except AssertionError:
            diff = ((end_idx_emg-start_idx_emg)+1)-np.size(line2.get_ydata())
            end_idx_emg = end_idx_emg-diff

        line2.set_ydata(long_emg[start_idx_emg:end_idx_emg+1])

    if long_v is not None:
        v_idx, = np.where(np.logical_and(long_v_t>=start_trace, long_v_t<=end_trace))
        # start_idx_v = np.where(v_t>=start_trace)[0][0]
        # end_idx_v = np.where(v_t<=end_trace)[0][-1]

        try:
            assert np.size(line3.get_xdata()) == np.size(v_idx)
            y_update = long_v[v_idx]
            # assert np.size(line3.get_ydata()) == (end_idx_v-start_idx_v)+1
        except AssertionError:
            y_update = np.empty(np.size(line3.get_xdata()))
            y_update[:] = np.nan
            if len(v_idx) != 0:
                y_update[:len(v_idx)] = long_v[v_idx]            

        line3.set_ydata(y_update)

    for i,m in enumerate(markers):
        if i == 1:
            m.set_xdata([int(this_epoch_t/epochlen),int(this_epoch_t/epochlen)])
        else:
            m.set_xdata([this_epoch_t,this_epoch_t])
    fig2.axes[0].set_xlim([this_epoch_t-600, this_epoch_t+600])
    fig2.axes[1].set_xlim([this_epoch_t-600, this_epoch_t+600])
    line4.set_xdata([this_epoch_t,this_epoch_t])
    line5.set_xdata([this_epoch_t,this_epoch_t])
    fig1.canvas.draw()
    fig2.canvas.draw()
def make_marker(fig, x, epochlen):
    markers = []
    for i,a in enumerate(fig.axes):
        ylims = list(a.get_ylim())
        xlims = list(a.get_xlim())
        if i == 1:
            marker, = a.plot([x/epochlen, x/epochlen], ylims, color = 'k')
        else:
            marker, = a.plot([x, x], ylims, color = 'k')
        a.set_ylim(ylims)
        markers.append(marker)
    return markers

def load_video(d, this_timestamp):
    print('Loading video now, this might take a second....')
    cap = {}
    fps = {}
    these_ts_files = np.unique(this_timestamp['Filename'])
    for ts in these_ts_files:
        v = get_videofn_from_csv(d, ts)
        print('Loading '+v+'...')
        cap[v] = cv2.VideoCapture(v)
        fps[v] = cap[v].get(cv2.CAP_PROP_FPS)
    return cap, fps

def timestamp_extracting(timestamp_file):
    ## Dealing with old Bonsai files is deprecated
    ts_datestring = time.ctime(os.path.getmtime(timestamp_file))
    timestamp_df = pd.read_csv(timestamp_file, header=None) 
    timestamp_df.columns = ['Timestamps']
    timestamp_df['Filename'] = timestamp_file

    ts_format1 = '%Y-%m-%dT%H:%M:%S.%f'
    ts_format2 = '%a %b %d %H:%M:%S %Y'
    short_ts = [x[:-6] for x in list(timestamp_df['Timestamps'].loc[~timestamp_df['Timestamps'].isnull()])]
    
    if datetime.strptime(short_ts[-1][:-1], ts_format1) > datetime.strptime(ts_datestring, ts_format2):
        time_adjust = datetime.strptime(short_ts[-1][:-1], ts_format1)-datetime.strptime(ts_datestring, ts_format2)
    else:
        time_adjust = datetime.strptime(ts_datestring, ts_format2)-datetime.strptime(short_ts[-1][:-1], ts_format1)
    try:
        assert time_adjust.days == 0
    except AssertionError:
        print('There is a problem with the timestamp adjustment')
        sys.exit()
    print(time_adjust.total_seconds())
    if len(short_ts) > 0:
        timestamp_df['Timestamps'] = [datetime.strptime(short_ts[i][:-1], ts_format1)-time_adjust for i in np.arange(len(short_ts))]

    return timestamp_df

def initialize_vid_and_move(d, a, acq_start, acq_len):
    if d['vid']:
        video_list = glob.glob(os.path.join(d['video_dir'], '*.mp4'))
        if len(video_list) == 0:
            video_list = glob.glob(os.path.join(d['video_dir'], '*.avi'))
        if len(video_list) == 0:
            print('No videos found! Please check directory')
            sys.exit()
        video_list = sort_files(video_list, d['basename'], d['csv_dir'])
        if d['Acquisition'].index(int(a)) == 0:
            this_video = video_list[0]
        else:
            timestamp_list = glob.glob(os.path.join(d['csv_dir'], '*timestamp*'))
            if len(timestamp_list) == 0:
                print('No timestamp files found! Please check directory')
                sys.exit()
            timestamp_list = sort_files(timestamp_list, d['basename'], d['csv_dir'])
            first_ts = [datetime.strptime(pd.read_csv(
                t, header = None).iloc[0][0][:-7], '%Y-%m-%dT%H:%M:%S.%f') for t in timestamp_list]
            acq_idx, = np.where([(acq_start > first_ts[ii]) & 
                (acq_start < first_ts[ii+1]) for ii in range(len(first_ts)-1)])
            this_video = video_list[acq_idx[0]]
    else:
        this_video = None
        print('no video available')
    if d['movement']:
        movement_df = pd.read_pickle(os.path.join(d['savedir'], 'All_movement.pkl'))
        start_ts = acq_start
        end_ts = start_ts+timedelta(seconds=acq_len)
        move_idx, = np.where(
            (movement_df['Timestamps'] < end_ts) & (movement_df['Timestamps'] > start_ts))
        this_motion = movement_df.iloc[move_idx]
        v = movement_processing(this_motion)

    else:
        v = None
        this_motion = None
    return this_video, v, this_motion


def get_ThD(this_eeg, fsd):
    Pxx, freqs, bins = my_specgram(this_eeg, Fs = fsd)
    delta_band = np.sum(Pxx[np.where(np.logical_and(freqs>=1,freqs<=4))],axis = 0)
    theta_band = np.sum(Pxx[np.where(np.logical_and(freqs>=5,freqs<=8))],axis = 0)

    return theta_band/delta_band

def movement_extracting(movement_file, d):
    if d['DLC']:
        movement_df_full = pd.read_csv(movement_file)
        movement_df = movement_df_full[[d['DLC Label']+'_x', d['DLC Label']+'_y', d['DLC Label']+'_likelihood']]
        movement_df.columns = ['X', 'Y', 'Likelihood']
    else:
        movement_df = pd.read_csv(movement_file, header = None)
        if len(movement_df.columns) == 2:
            movement_df.columns = ['X','Y']
            if movement_df['X'].iloc[0] == 'X':
                movement_df = movement_df.drop(0)
                movement_df = movement_df.reset_index(drop = True)
        else:
            movement_df.columns = ['Timestamps', 'X','Y']
            ts_format = '%Y-%m-%dT%H:%M:%S.%f'
            short_ts = [x[:-6] for x in list(movement_df['Timestamps'])]
            movement_df['Timestamps'] = [datetime.strptime(short_ts[i][:-1], ts_format) for i in np.arange(len(short_ts))]
    movement_df['Filename'] = movement_file
    return movement_df

def movement_processing(this_motion, binsize = 4):
    this_motion['X'] = this_motion['X'].fillna(0)
    this_motion['Y'] = this_motion['Y'].fillna(0)
    try:
        t_vect = this_motion['Timestamps']-this_motion['Timestamps'].iloc[0]
    except IndexError:
        t = np.arange(0, 3600, binsize)
        v = [np.nan for i in range(len(t))]
        v = np.vstack([v,t])
        return v
    t_vect = t_vect.dt.total_seconds().values
    # t_vect = [t_vect.iloc[i].total_seconds() for i in range(len(t_vect))]
    bins = np.arange(0, t_vect[-1]+binsize, binsize)
    # dx = []
    # dy = []
    # t = []
    # ts = []
    idxs = [np.where(np.logical_and(t_vect>=bins[i], t_vect<bins[i+1]))[0] 
                for i in np.arange(0, np.size(bins)-1)]
    
    dx = [this_motion['X'].iloc[ii[-1]] - this_motion['X'].iloc[ii[0]] 
                for ii in idxs if len(ii) > 0]
    dy = [this_motion['Y'].iloc[ii[-1]] - this_motion['Y'].iloc[ii[0]] 
                for ii in idxs if len(ii) > 0]
    t = [t_vect[ii[-1]] for ii in idxs if len(ii) > 0]
    v = np.sqrt((np.square(dx) + np.square(dy)))
    v = np.vstack([v,t])

    # for i in np.arange(0, np.size(bins)-1)]
    # for i in np.arange(0, np.size(bins)-1):
    #     idxs, = np.where(np.logical_and(t_vect>=bins[i], t_vect<bins[i+1]))
    #     temp_x = list(this_motion['X'].iloc[idxs])
    #     dx.append(int(float(temp_x[-1]))-int(float(temp_x[0])))
    #     temp_y = list(this_motion['Y'].iloc[idxs])
    #     dy.append(int(float(temp_y[-1]))-int(float(temp_y[0])))
    #     t.append(t_vect[idxs[-1]])
    #     ts.append(this_motion['Timestamps'].iloc[idxs[-1]])

    return v

def get_movement_segs(movement_df, time_window):
    t_vect = (movement_df['Timestamps'] - movement_df['Timestamps'].iloc[0]).dt.total_seconds()
    vs = []
    if time_window.ndim == 2:
        for w in time_window:
            this_motion = movement_df.loc[(t_vect >= w[0]) & (t_vect < w[1])]
            vs.append(movement_processing(this_motion))
        return vs
    else:
        this_motion = movement_df.loc[(t_vect >= time_window[0]) & (t_vect < time_window[1])]
        v = movement_processing(this_motion)
        return v
    

def prepare_feature_data(FeatureDict, movement_flag, smooth = False):
    del FeatureDict['animal_name']
    if movement_flag:
        FeatureDict = adjust_movement(FeatureDict, movement_flag)
    if smooth:
        FeatureList = []
        for f in FeatureDict.keys():
            FeatureList_smoothed.append(signal.medfilt(FeatureDict[f], 5))
    else:
        FeatureList = list(FeatureDict.values())
    Features = np.column_stack((FeatureList))
    Features = np.nan_to_num(Features)
    return Features

def build_feature_dict(eeg_df, fsd, epochlen, normVal = None):
    FeatureDict = {}
    acq_len = len(eeg_df)/fsd
    num_epochs = acq_len/epochlen

    freq_dict = freq_dict = {'Delta': [0.5, 4], 
                            'Theta':[5, 8],
                            'Alpha': [8, 12],
                            'BroadTheta':[2, 16],
                            'Fire': [4, 20]}
    print('Generating EMG vectors...')
    if 'EMG' in eeg_df.columns:
        FeatureDict['EMGvar'], EMGmax, EMGmean = generate_signal(eeg_df['EMG'], epochlen, fsd)

    print('Generating EEG vectors...')
    for ii, c in enumerate([i for i in eeg_df.columns if 'EEG' in i]):
        FeatureDict[c+'var'], EEGmax, EEGmean = generate_signal(eeg_df[c], epochlen, fsd)
        power_dict = bandPower(eeg_df[c], fsd, freq_dict = freq_dict, minfreq = 0.5, maxfreq = 16)
        epoch_bins = np.arange(0, acq_len, epochlen)
        epoch_idx = [np.where(np.logical_and(power_dict['Bins']>=i, power_dict['Bins']<i+4))[0] for i in epoch_bins]
        
        for band in list(freq_dict.keys()):
            FeatureDict[c+band] = [np.median(power_dict[band][i]) for i in epoch_idx]/normVal[ii]
            for n in np.where(np.isnan(FeatureDict[c+band]))[0]:
                if n < np.size(FeatureDict[c+band])-1:
                    FeatureDict[c+band][n] = FeatureDict[c+band][n+1]
                else:
                    FeatureDict[c+band][n] = FeatureDict[c+band][n-1]
            # FeatureDict['EEG'+band] = FeatureDict['EEG'+band]/normVal
            assert np.size(FeatureDict[c+band]) == num_epochs

        FeatureDict[c+'nb'] = FeatureDict[c+'Theta'] / FeatureDict[c+'BroadTheta'] # narrow-band theta
        # # delt_thet = EEGdelta / EEGtheta # ratio; esp. important
        FeatureDict[c+'thet_delt'] = FeatureDict[c+'Theta'] / FeatureDict[c+'Delta']


        # frame shifting
        FeatureDict[c+'delta_post'], FeatureDict[c+'delta_pre'] = post_pre(FeatureDict[c+'Delta'], 
            FeatureDict[c+'Delta'])
        FeatureDict[c+'theta_post'], FeatureDict[c+'theta_pre'] = post_pre(FeatureDict[c+'Theta'], 
            FeatureDict[c+'Theta'])
        FeatureDict[c+'delta_post2'], FeatureDict[c+'delta_pre2'] = post_pre(FeatureDict[c+'delta_post'], 
            FeatureDict[c+'delta_pre'])
        FeatureDict[c+'theta_post2'], FeatureDict[c+'theta_pre2'] = post_pre(FeatureDict[c+'theta_post'], 
            FeatureDict[c+'theta_pre'])
        FeatureDict[c+'delta_post3'], FeatureDict[c+'delta_pre3'] = post_pre(FeatureDict[c+'delta_post2'], 
            FeatureDict[c+'delta_pre2'])
        FeatureDict[c+'theta_post3'], FeatureDict[c+'theta_pre3'] = post_pre(FeatureDict[c+'theta_post2'], 
            FeatureDict[c+'theta_pre2'])
        FeatureDict[c+'nb_post'], FeatureDict[c+'nb_pre'] = post_pre(FeatureDict[c+'nb'], 
            FeatureDict[c+'nb'])
   
    return FeatureDict

def adjust_movement(FeatureDict, epochlen = 4):
    v = FeatureDict['Velocity']
    if np.size(v) > 900:
        v_reshape = np.reshape(v, (-1,epochlen))
        mean_v = np.mean(v_reshape, axis = 1)
        mean_v[np.isnan(mean_v)] = 0
    elif np.size(v) < 900:
        diff = 900 - np.size(v)
        nans = np.empty(diff)
        nans[:] = 0
        mean_v = np.concatenate((v, nans))
    else:
        mean_v = v

    mean_v[np.isnan(mean_v)] = 0
    FeatureDict['Velocity'] = mean_v
    return FeatureDict

def model_feature_importance(filename_sw):
    with open(filename_sw, 'r') as f:
        d = json.load(f)

    emg_flag = int(d['emg'])
    movement_flag = int(d['movement'])
    model_dir = str(d['model_dir'])
    mod_name = str(d['mod_name'])

    if emg_flag:
        jobname = mod_name + '_EMG'
        print("EMG flag on")
    else:
        x_features.remove('EMG')
        jobname = mod_name + '_no_EMG'
        print('Just so you know...this model has no EMG')
    if movement_flag:
        jobname = jobname + '_movement'
    else:
        jobname = jobname + '_no_movement'
    jobname = jobname + '.joblib'

    clf = joblib.load(os.path.join(model_dir, jobname))
    Sleep_Model = np.load(file = model_dir + mod_name + '_model.pkl', allow_pickle = True)
    del Sleep_Model['State']
    del Sleep_Model['animal_name']

    fig,ax = plt.subplots(figsize = (14,8))
    y = clf.feature_importances_
    x = np.arange(len(y))
   
    ax.bar(x, y, color = 'k')
    ax.set_xticks(x)
    ax.set_xticklabels(list(Sleep_Model.columns), rotation = 45)
    ax.set_xlabel('Features')
    ax.set_ylabel('Feature Importance')
    sns.despine()
    return fig


def rename_DLC_csvs(csv_dir, basename):
    DLC_coords_files = glob.glob(os.path.join(csv_dir, 'Coord*.csv'))
    for f in DLC_coords_files:
        fname = os.path.split(f)[1]
        substring_1 = 'Coord'
        idx = fname.find('DLC')
        fname_new = fname[:idx]
        fname_new = fname_new.replace(substring_1, "")
        insert_idx = fname_new.find(basename) + len(basename)
        fname_change = fname_new[:insert_idx]+'_motion'+fname_new[insert_idx:]+'.csv'
        os.rename(f,os.path.join(csv_dir, fname_change))

def DLC_check_fig(csv_file):
    coords_df = pd.read_csv(csv_file)
    color_dict = {}
    labels = ['center', 'ear1', 'ear2', 'nose', 'baseoftail']
    color_dict['center'] = '#fffd01'
    color_dict['ear1'] = '#7e1e9c'
    color_dict['ear2'] = '#cb416b'
    color_dict['nose'] = '#2000b1'
    color_dict['baseoftail'] = '#f0944d'

    fig, ax = plt.subplots(nrows = len(labels), figsize = (15, 8))
    vel_dict = {}
    x = np.linspace(0, 3600, len(coords_df[labels[0]+'_x']))
    bins = np.arange(0, 3601)
    for i,l in enumerate(labels):
        dx = []
        dy = []
        for ii in np.arange(0, np.size(bins)-1):
            idxs, = np.where(np.logical_and(x>=bins[ii], x<bins[ii+1]))
            temp_x = list(coords_df[l+'_x'].iloc[idxs])
            dx.append(temp_x[-1]-temp_x[0])
            temp_y = list(coords_df[l+'_y'].iloc[idxs])
            dy.append(temp_y[-1]-temp_y[0])
        vel_dict[l] = np.sqrt((np.square(dx) + np.square(dy)))
        ax[i].plot(bins[1:]/60, vel_dict[l], color = color_dict[l], label = l)
        ax[i].set_title(l)
        ax[i].set_xlim([0,60])
        ax[i].set_xticks(np.arange(0, 60))
        ax[i].set_yticklabels([])
        sns.despine()

    fig.tight_layout()
    
def transfer_DLC_files(transfer_directory, basenames, DLC_model_dir):
    for b in basenames:
        try:
            os.rename(os.path.join(transfer_directory, b,b+'_csv'), os.path.join(transfer_directory, b,b+'_csv_old2'))
        except OSError:
            pass
        try:
            os.mkdir(os.path.join(transfer_directory, b, 'DLC_Outputs'))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(transfer_directory, b, b+'_csv'))
        except FileExistsError:
            pass
        files_to_move = []
        coord_files = []
        for i in ['up_day_t/Test_day_updated-Samarth-2023-05-17', 'up_night_t/Test_night_updated-Samarth-2023-05-17']:
            files_to_move.append(glob.glob(os.path.join(DLC_model_dir, i, 'Testing', b+'*labeled.mp4')))
            files_to_move.append(glob.glob(os.path.join(DLC_model_dir, i, 'Testing', b+'*.pickle')))
            files_to_move.append(glob.glob(os.path.join(DLC_model_dir, i, 'Testing', b+'*.h5')))
            files_to_move.append(glob.glob(os.path.join(DLC_model_dir, i, 'Testing', b+'*.csv')))
            coord_files.append(glob.glob(os.path.join(DLC_model_dir, i, 'Testing/coords_csv', '*'+b+'*.csv')))
        files_to_move = np.concatenate(files_to_move)
        coord_files = np.concatenate(coord_files)
        timestamp_files = glob.glob(os.path.join(transfer_directory, b, b+'_csv_old2', '*timestamp*'))
        for f in timestamp_files:
            directory, fname = os.path.split(f)
            shutil.move(f, os.path.join(transfer_directory, b, b+'_csv', fname))
        for f in files_to_move:
            directory, fname = os.path.split(f)
            shutil.move(f, os.path.join(transfer_directory, b, 'DLC_Outputs', fname))
        for f in coord_files:
            directory, fname = os.path.split(f)
            shutil.move(f, os.path.join(transfer_directory, b, b+'_csv'))
        rename_DLC_csvs(os.path.join(transfer_directory, b, b+'_csv'), b)
    print('Done :)')

def get_videofn_from_csv(d, csv_filename):
    str_idx1 = csv_filename.find('timestamp')+len('timestamp')
    str_idx2 = csv_filename.find('.csv')
    num = csv_filename[str_idx1:str_idx2]
    fn = os.path.split(csv_filename)[1]
    v = os.path.join(d['video_dir'], d['basename']+num+'.mp4')
    return v
def define_microarousals(sleep_states, epoch_len):
    wake_idx = PKA.find_continuous(sleep_states, [1,4])
    for w in wake_idx:
        if len(w)*epoch_len <= 16:
            sleep_states[w] = 5
    return sleep_states
def get_eeg_segment(basename, time_window):
    file_num, sec_in = divmod(time_window[0], 3600)
    time_diff = time_window[-1]-time_window[0]
    eeg_file = get_eeg_file(basename, int(file_num))
    eeg = np.load(eeg_file)
    eeg_t = np.linspace(0, 3600, np.size(eeg))
    seg_idx, = np.where(np.logical_and(eeg_t>=sec_in, eeg_t<sec_in+time_diff))
    eeg_seg = eeg[seg_idx]
    return eeg_seg

def get_eeg_file(basename, file_num):
    extracted_dir = os.path.join('/Volumes/yaochen/Active/Lizzie/FLP_data', basename, basename+'_extracted_data')
    downsampled_files = glob.glob(os.path.join(extracted_dir, 'downsampEEG*_hr0.npy'))
    acqs = np.sort([int(d[d.find('Acq')+3:d.find('_hr0')]) for d in downsampled_files])
    this_file = os.path.join(extracted_dir, 'downsampEEG_Acq'+str(acqs[file_num])+'_hr0.npy')
    return this_file

def sort_timestamp_files(timestamp_dir):
    timestamp_list = glob.glob(os.path.join(timestamp_dir, '*timestamp*'))
    first_ts = [datetime.strptime(pd.read_csv(
        t, header = None).iloc[0][0][:-7], '%Y-%m-%dT%H:%M:%S.%f') for t in timestamp_list]
    sorting_idx = np.argsort(first_ts)
    file_labels = []
    for t in timestamp_list:
        idx1 = t.find('timestamp')+9
        idx2 = t.find('.csv')
        file_labels.append(int(t[idx1:idx2]))
    sorted_file_labels = np.asarray(file_labels)[sorting_idx]

    return sorted_file_labels

def sort_files(file_list, basename, timestamp_dir):
    sorted_file_labels = sort_timestamp_files(timestamp_dir)
    fn_only = [os.path.splitext(os.path.basename(l))[0] for l in file_list]
    ext = os.path.splitext(file_list[0])[1]
    if ext == '.mp4':
        file_nums = [int(i[i.find(basename)+len(basename):]) for i in fn_only]
    if ext == '.csv':
        if 'motion' in file_list[0]:
            file_nums = [int(i[i.find('motion')+len('motion'):]) for i in fn_only]
        elif 'timestamp' in file_list[0]:
            file_nums = [int(i[i.find('timestamp')+len('timestamp'):]) for i in fn_only]
        else:
            print('I do not know what type of files these are....')
    if len(file_nums) != len(sorted_file_labels):
        print('You have timestamp files for the following numbers but they do not appear in the given list:')
        print([n for n in sorted_file_labels if n not in file_nums], sep="\n")
        sys.exit()

    ordered_idx = [file_nums.index(f) for f in sorted_file_labels]
    sorted_files = [file_list[idx] for idx in ordered_idx]

    return sorted_files
def get_total_power(this_eeg, fsd):
    Pxx, freqs, bins = my_specgram(this_eeg, Fs = fsd)
    freq_res = freqs[1]-freqs[0]
    total_power = simps(Pxx, dx=freq_res, axis = 0)
    return total_power

def get_AcqStart(d, a, acq_len):
    if len(glob.glob(os.path.join(d['rawdat_dir'], 'trigger_times.mat'))) != 0:
        trigger_times = {}
        io.loadmat(glob.glob(os.path.join(d['rawdat_dir'], 'trigger_times.mat'))[0], 
            mdict=trigger_times)
        trigger_times = trigger_times['trigger_times'][0]
        acq_start = datetime(*[int(ii) for ii in trigger_times[d['Acquisition'].index(int(a))][0]])
    else:
        AD_file = os.path.join(d['rawdat_dir'], 'AD' + str(d['EEG channel'][0]) + '_'+str(a)+'.mat')
        EEG_datestring = time.ctime(os.path.getmtime(AD_file))
        ts_format = '%a %b %d %H:%M:%S %Y'
        EEG_datetime = datetime.strptime(EEG_datestring, ts_format)
        acq_start = EEG_datetime-timedelta(seconds=acq_len)
    return acq_start

def pulling_timestamp(timestamp_df, acq_start, this_eeg, fsd):
    acq_len = int(np.size(this_eeg)/fsd)
    start_ts = acq_start
    end_ts = start_ts+timedelta(seconds=acq_len)
    ts_idx, = np.where(np.logical_and(timestamp_df['Timestamps'] < end_ts, timestamp_df['Timestamps'] > start_ts))
    this_timestamp = timestamp_df.iloc[ts_idx]
    offset_times = this_timestamp['Timestamps']-this_timestamp['Timestamps'].iloc[0]
    this_timestamp['Offset_Time'] = [offset_times.iloc[i].total_seconds() for i in range(len(offset_times))]

    return this_timestamp

def print_instructions():
    print('''\

                            .--,       .--,
                           ( (  \.---./  ) )
                            '.__/o   o\__.'
                               {=  ^  =}
                                >  -  <
        ____________________.""`-------`"".________________________

                              INSTRUCTIONS

        Welcome to Sleep Wake Scoring!

        The figure you're looking at consists of 3 plots:
        1. The spectrogram for the hour you're scoring
        2. The random forest model's predicted states
        3. The binned motion for the hour

        TO CORRECT BINS:
        - click once on the middle figure to select the start of the bin you want to change
        - then click the last spot of the bin you want to change
        - switch to terminal and type the state you want that bin to become

        VIDEO / RAW DATA:
        - if you hover over the motion figure you enter ~~ movie mode ~~
        - click on that figure where you want to pull up movie and the raw trace for
            the 4 seconds before, during, and after the point that you clicked

        CURSOR:
        - because you have to click in certain figures it can be annoying to line up your mouse
            with where you want to inspect
        - while selected in the scoring figure (called Figure 2) press 'l' (as in Lizzie) to toggle a black line across each plot
        - this line will stay there until you press 'l' again, then it will erase and move
        - adjust until you like your location, then click to select a bin or watch a movie

        EXITING SCORING:
        - are you done correcting bins?
        - are you sure?
        - are you going to come to me/clayton/lizzie and ask how you 'go back' and 'change a few more bins'?
        - think for a second and then, when you're sure, press 'd'
        - it will then ask you if you want to save your states and/or update the random forest model
            - choose wisely

        NOTES:
        - all keys pressed should be lowercase. don't 'shift + d'. just 'd'.
        - the video window along with the raw trace figure will remain up and update when you click a new bin
            don't worry about closing them or quitting them, it will probably error if you do.
        - slack me any errors if you get them or you have ideas for new functionality/GUI design
            - always looking to stay ~fresh~ with those ~graphics~
        - if something isn't working, make sure you're on Figure 2 and not the raw trace/terminal/video
        - plz don't toggle line while in motion axes, it messes up the axes limits, not sure why, working on it

        coming soon to sleep-wake code near you:
        - coding the state while you're slected in the figure, so you don't have to switch to terminal
        - automatically highlighting problem areas where the model isn't sure or a red flag is raised (going wake/rem/wake/rem)
        - letting you choose the best fitting model before you fix the states to limit the amont of corrections


        ANOUNCEMENTS:
        - if you're trying to code each bin individually (a.k.a. when it asks you if you want to correct the model you say 'no')
            it doesn't save afterward yet. you will have to manually save it after you're done for the time being

                                               ''')

### - - - - - - - - - - - - - - - - DEPRECATED FUNCTIONS - - - - - - - - - -  - - ####





