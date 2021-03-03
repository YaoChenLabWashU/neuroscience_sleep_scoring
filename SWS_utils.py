import os
import numpy as np
import sys
from scipy.integrate import simps
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patch
import copy as cp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from joblib import dump, load
import pandas as pd
import cv2
import math
from pylab import *
from matplotlib import *


def generate_signal(downsamp_signal, epochlen, fs): # fs = fsd here
    # mean of 4 seconds
    normmean = np.mean(downsamp_signal)
    normstd = np.std(downsamp_signal)
    binl = epochlen * fs  # bin size in array slots | number of points in an epoch; a bin == an epoch
    sig_amp = np.zeros(int(np.size(downsamp_signal) / (epochlen * fs)))
    sig_mean = np.zeros(int(np.size(downsamp_signal) / (epochlen  * fs)))
    for i in np.arange(np.size(sig_amp)):
        sig_amp[i] = np.var(downsamp_signal[epochlen  * fs * (i):(epochlen  * fs * (i + 1))])
        sig_mean[i] = np.mean(np.abs(downsamp_signal[epochlen  * fs * (i):(epochlen  * fs * (i + 1))]))
    sig_amp = (sig_amp - normmean) / normstd # normalization
    # we do not normalize mean (for some reason)

    # TODO: for weird reason they are not in the same loop...; can potentially combine
    sig_max = np.zeros(int(np.size(downsamp_signal) / (epochlen  * fs)))
    for i in np.arange(np.size(sig_max)):
        sig_max[i] = np.max(downsamp_signal[epochlen  * fs * (i):(epochlen  * fs * (i + 1))])
    sig_max = (sig_max - np.average(sig_max)) / np.std(sig_max)
    
    return sig_amp, sig_max, sig_mean

def bandPower(low, high, downsamp_EEG, epochlen, fsd):
	win = epochlen * fsd # window == bin
	EEG = np.zeros(int(np.size(downsamp_EEG)/(epochlen*fsd)))
	EEGreshape = np.reshape(downsamp_EEG,(-1,fsd*epochlen)) # funky
	freqs, psd = signal.welch(EEGreshape, fsd, nperseg=win, scaling='density') # for each freq, have a power value
	idx_min = np.argmax(freqs > low) - 1
	idx_max = np.argmax(freqs > high) - 1
	idx = np.zeros(dtype=bool, shape=freqs.shape)
	idx[idx_min:idx_max] = True
	EEG = simps(psd[:,idx], freqs[idx])
	return EEG, idx

def normalize(toNorm):
	norm = (toNorm - np.average(toNorm))/np.std(toNorm)
	return norm

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

def plot_spectrogram(ax, eegdat, fsd):
    ax.set_title('Spectrogram w/ EMG')
    window_length = 5 # n seconds in windowing segments
    noverlap = 4.9 # step size in sec
    dt = 1/fsd
    t_elapsed = eegdat.shape[0]/fsd
    t = np.arange(0.0, t_elapsed, dt)
    noverlap = noverlap * fsd
    NFFT = window_length * fsd
    minfreq = 1 # min freq in Hz
    maxfreq = 16 # max freq in Hz
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    # the minfreq and maxfreq args will limit the frequencies
    Pxx, freqs, bins, im = my_specgram(eegdat, ax, NFFT=NFFT, Fs=fsd, noverlap=noverlap,
                                cmap=cm.get_cmap('jet'), minfreq = minfreq, maxfreq = maxfreq,
                                xextent = (0,np.int(t_elapsed)) )

def my_specgram(x, ax, NFFT=400, Fs=200, Fc=0, detrend=mlab.detrend_none,
             window=mlab.window_hanning, noverlap=200,
             cmap=None, xextent=None, pad_to=None, sides='default',
             scale_by_freq=None, minfreq = None, maxfreq = None, **kwargs):
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
    im = ax.imshow(Z, cmap, extent=extent, **kwargs)
    ax.axis('auto')

    return Pxx, freqs, bins, im

def plot_predicted(ax, Predict_y, clf, Features):
    ax.set_title('Predicted States')
    for state in np.arange(np.size(Predict_y)):
        if Predict_y[state] == 0:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = 'green')
            ax.add_patch(rect7)
        elif Predict_y[state] == 2:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = 'blue')
            ax.add_patch(rect7)
        elif Predict_y[state] == 5:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = 'red')
            ax.add_patch(rect7)
        elif Predict_y[state] == 4:
            rect7 = patch.Rectangle((state, 0), 3.8, height = 1, color = '#a8a485')
            ax.add_patch(rect7)
    ax.set_ylim(0.3, 1)
    ax.set_xlim(0, 900)
    predictions = clf.predict_proba(Features)
    confidence = np.max(predictions, 1)
    ax.plot(confidence, color = 'k')

def create_prediction_figure(Predict_y, clf, Features, fs, eeg):
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, figsize = (11, 6))
    plot_spectrogram(ax1, eeg, fs)
    plot_predicted(ax2, Predict_y, clf, Features)
    fig.tight_layout()
    return fig, ax1, ax2, ax3

def update_sleep_model(model_dir, mod_name, df_additions):
    try:
        Sleep_Model = np.load(file = model_dir + mod_name + '_model.pkl', allow_pickle = True)
        Sleep_Model = Sleep_Model.append(df_additions, ignore_index = True)
    except FileNotFoundError:
        print('no model created...I will save this one')
        df_additions.to_pickle(model_dir + mod_name + '_model.pkl')
        Sleep_Model = df_additions
    Sleep_Model.to_pickle(model_dir + mod_name + '_model.pkl')
    return Sleep_Model

def load_joblib(FeatureList, emg_flag, mod_name):
    x_features = cp.deepcopy(FeatureList)
    [x_features.remove(i) for i in ['Animal_Name', 'State']]
    jobname =''

    if emg_flag:
        jobname = mod_name + '_EMG.joblib'
        print("EMG flag on")
    else:
        x_features.remove('EMG')
        jobname = mod_name + '_no_EMG.joblib'
        print('Just so you know...this model has no EMG')
    return jobname, x_features

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

    Satisfaction = input('Satisfied?: ')
    if Satisfaction == 'y':
        # clf = random_forest_classifier(Sleep_Model[x_features].apply(pd.to_numeric).values,
        #                                Sleep_Model['State'].apply(pd.to_numeric).values)
        # print("Train Accuracy :: ", accuracy_score(Sleep_Model['State'].apply(pd.to_numeric).values,
        #                                            clf.predict(Sleep_Model[x_features].apply(pd.to_numeric).values)))
        dump(clf, model_dir + jobname)


def pull_up_movie(start, end, vid_file, epochlen):
    cap = cv2.VideoCapture(vid_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('Pulling up video ....')
    print('starting at ' + str(start/fps) + ' seconds')
    if not cap.isOpened():
        print("Error opening video stream or file")
    score_win = np.arange(start+fps*1*epochlen, start+fps*2*epochlen)
    for f in np.arange(start, end):
        cap.set(1, f)
        ret, frame = cap.read()
        if ret:
            if f in score_win:
                cv2.putText(frame, "SCORE WINDOW", (50, 105), cv2.FONT_HERSHEY_PLAIN, 4, (225, 0, 0), 2)
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
def pull_up_raw_trace(ax1, ax2, ax3,ax4, emg, start, end, realtime, 
    this_eeg, fsd, LFP_ylim, delt, thet, epochlen, this_emg):
    print('pull up the second figure for that bin - maybe. Option to click through a few bins around it?')
    x = (end - start)
    length = np.arange(int(end / x - start / x))
    bottom = np.zeros(int(end / x - start / x))

    assert np.size(delt) == np.size(this_eeg) == np.size(thet)

    line1 = plot_LFP(start, end, ax1, this_eeg, realtime, fsd, LFP_ylim, epochlen)
    line2 = plot_delta(delt, start, end, fsd, ax2, epochlen, realtime)
    line3 = plot_theta(ax3, start, end, fsd, thet, epochlen, realtime)

    if not emg:
        ax4.text(0.5, 0.5, 'There is no EMG')
        line4 = plt.plot([0,0], [1,1], linewidth = 0, color = 'w')
    else:
        line4 = plot_EMG(ax4, length, bottom, this_emg, epochlen, x, start, end, realtime, fsd)

    return line1, line2, line3, line4

def plot_delta(delt, start, end, fsd, ax, epochlen, realtime):
    line2, = ax.plot(realtime[start:end], delt[start:end])
    ax.set_xlim(start/fsd, end/fsd)
    ax.set_ylim(-2, 2)
    bottom_2 = ax.get_ylim()[0]
    rectangle_2 = patch.Rectangle((start/fsd+epochlen,bottom_2),epochlen,height=float(-bottom_2/5))
    ax.add_patch(rectangle_2)
    ax.set_title('Delta power (0.5 - 4 Hz)')
    return line2

def plot_theta(ax, start, end, fsd, theta, epochlen, realtime):
    line3, = ax.plot(realtime[start:end], theta[start:end])
    ax.set_xlim(start/fsd, end/fsd)
    ax.set_ylim(-2,2)
    ax.set_title('Theta power (4 - 8 Hz)')
    bottom_3 = ax.get_ylim()[0]
    rectangle_3 = patch.Rectangle((start/fsd+epochlen, bottom_3), epochlen, height = -bottom_3 / 5)
    ax.add_patch(rectangle_3)
    return line3

def plot_LFP(start, end, ax, this_eeg, realtime, fsd, LFP_ylim, epochlen):
    line1, = ax.plot(realtime[start:end], this_eeg[start:end])
    ax.set_xlim(start/fsd, end/fsd)
    ax.set_title('LFP')
    ax.set_ylim(-LFP_ylim, LFP_ylim)
    bottom = -LFP_ylim
    rectangle = patch.Rectangle((start/fsd+epochlen, bottom),epochlen,height=-bottom/5)
    ax.add_patch(rectangle)
    return line1

def plot_EMG(ax, length, bottom, this_emg, epochlen, x, start, end, realtime, fsd):
    # anything with EMG will error
    line4, = ax.plot(realtime[start:end], this_emg[start:end], color = 'r')    
    ax.set_title('EMG Amplitde')
    ax.set_xlim(start/fsd, end/fsd)
    ax.set_ylim(0, 0.3)
    top = ax.get_ylim()[1]
    rectangle_4 = patch.Rectangle((start/fsd+epochlen, top), epochlen, height = -top / 5, color = 'r')
    ax.add_patch(rectangle_4)
    return line4

def clear_bins(bins, ax2):
    for b in np.arange(bins[0], bins[1]):
        b = math.floor(b)
        location = b
        rectangle = patch.Rectangle((location, 0), 1.5, height = 2, color = 'white')
        ax2.add_patch(rectangle)
def correct_bins(start_bin, end_bin, ax2, new_state):
    for b in np.arange(start_bin, end_bin):
        b = math.floor(b)
        location = b
        color = 'white'
        if new_state == 1:
            color = 'green'
        if new_state == 2:
            color = 'blue'
        if new_state == 3:
            color = 'red'
        rectangle = patch.Rectangle((location, 0), 1.5, height = 2, color = color)
        print('loc: ', location)
        ax2.add_patch(rectangle)

def create_scoring_figure(extracted_dir, a, eeg, fsd):
    fig = plt.figure(constrained_layout=True, figsize = (11, 6))
    widths = [1]
    heights = [2,1,0.5]
    spec = fig.add_gridspec(ncols=1, nrows=3, width_ratios=widths, height_ratios=heights)
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1])
    ax3 = fig.add_subplot(spec[2])

    #fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (11, 6))
    plot_spectrogram(ax1, eeg, fsd)
    ax2.set_ylim(0.3, 1)
    ax2.set_xlim(0, int(np.size(eeg)/fsd))
    ax3.set_xlim([0,1])
    ax3.set_ylim([0,1])
    rect = patch.Rectangle((0,0),1,1, color = 'k')
    ax3.add_patch(rect)
    ax3.text(0.5, 0.5, 'Click here for video', horizontalalignment='center', 
        verticalalignment='center', transform=ax3.transAxes, color = 'w', fontsize = 16)
    ax3.tick_params(axis='both',which='both',bottom=False,top=False,left = False, 
        labelbottom=False, labelleft = False)

    fig.show()
    fig.tight_layout()
    return fig, ax1, ax2

def update_raw_trace(line1, line2, line3, line4, marker, fig, fig2, start, end, 
    this_eeg, delt, thet, emg_flag, this_emg, realtime):
    line1.set_ydata(this_eeg[start:end])
    line2.set_ydata(delt[start:end])
    line3.set_ydata(thet[start:end])
    marker.set_xdata([realtime[end],realtime[end]])
    if emg_flag:
        line4.set_ydata(this_emg[start:end])
    else:
        line4.set_ydata([1,1])
    fig.canvas.draw()
    fig2.canvas.draw()
def make_marker(ax, end, realtime, fsd, epochlen):
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    marker, = ax.plot([realtime[end], realtime[end]], [ymin, ymax], color = 'k')
    return marker
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






