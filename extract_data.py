import numpy as np
import glob
import json
import scipy.io
import os
import scipy.signal as signal
import matplotlib.pyplot as plt
import os.path
import math
import psutil


def choosing_acquisition(filename_sw):
	with open(filename_sw, 'r') as f:
	       d = json.load(f)

	rawdat_dir = str(d['rawdat_dir'])
	fs = int(d['fs'])
	EEG_chan = str(d['EEG channel'])

	print(rawdat_dir)
	os.chdir(rawdat_dir)

	poss_files = glob.glob('AD'+EEG_chan+'*.mat')
	acq = []
	for ii in poss_files:
		print(ii)
		try:
			idx1 = ii.find('_')
			idx2 = ii.find('.mat')
			int(ii[idx1+1:idx2])

		except ValueError:
			continue 
		eeg = scipy.io.loadmat(ii)[ii[0:idx2]][0][0][0][0]
		acq_len = np.size(eeg)/fs
		print('This acquisition is ' + str(round(acq_len/60, 1)) + ' minutes')
		decision = input('Would you like to use this acquisition? (y/n)')

		if decision == 'y':
			acq.append(int(ii[idx1+1:idx2]))
		elif decision == 'n':
			print('Ok, not adding this one')
		else: 
			print('I did not understand that input. Please run this function again.')
			return
	print('Copy the values below to the Acquisition line in ' + filename_sw)
	print(sorted(acq))


def downsample_filter(filename_sw):
	with open(filename_sw, 'r') as f:
			d = json.load(f)

	rawdat_dir = str(d['rawdat_dir'])
	model_dir = str(d['model_dir'])
	animal = str(d['animal'])
	epochlen = int(d['epochlen'])
	fs = int(d['fs'])
	emg_flag = int(d['emg'])
	vid = int(d['vid'])
	EEG_chan = str(d['EEG channel'])
	EMG_chan = str(d['EMG channel'])
	acq = d['Acquisition']
	filt_high = int(d['Filter High'])
	filt_low = d['Filter Low']
	savedir = str(d['savedir'])
	fsd = int(d['fsd'])
	if not os.path.exists(savedir):
		os.mkdir(savedir)

	print(rawdat_dir)
	os.chdir(rawdat_dir)

	EEG_files = [glob.glob('AD'+EEG_chan+'_'+str(i)+'.mat') for i in acq]
	EEG_files = np.asarray(np.concatenate(EEG_files))

	EMG_files = [glob.glob('AD'+EMG_chan+'_'+str(i)+'.mat') for i in acq]
	EMG_files = np.asarray(np.concatenate(EMG_files))

	nyq = 0.5*fs
	N  = 3    # Filter order
	

	for fil in np.arange(np.size(acq)):
		Wn = [filt_low/nyq,filt_high/nyq] # Cutoff frequencies
		f_eeg = EEG_files[fil]
		f_emg = EMG_files[fil]
		eeg = scipy.io.loadmat(f_eeg)['AD'+EEG_chan+'_'+str(acq[fil])]
		eeg = eeg[0][0][0][0]
		B, A = signal.butter(N, Wn, btype='bandpass',output='ba')
		eegfilt = signal.filtfilt(B,A, eeg)
		acq_len = np.size(eegfilt)/fs
		new_len = acq_len*fsd
		if emg_flag == 1:
			emg = scipy.io.loadmat(f_emg)['AD'+EMG_chan+'_'+str(acq[fil])]
			emg = emg[0][0][0][0]
			Wn = [10/nyq] # Cutoff frequencies
			B, A = signal.butter(N, Wn, btype='highpass',output='ba')
			emgfilt = signal.filtfilt(B,A, emg)
			emg_downsamp = signal.resample(emgfilt, int(new_len))
			emg_abs = np.absolute(emg_downsamp)
		eeg_downsamp = signal.resample(eegfilt, int(new_len))
		np.save(os.path.join(savedir, 'downsampEEG_Acq'+str(acq[fil])), eeg_downsamp)
		np.save(os.path.join(savedir, 'downsampEMG_Acq'+str(acq[fil])), emg_abs)
def create_spectrogram(filename_sw):
	with open(filename_sw, 'r') as f:
		d = json.load(f)

	fsd = int(d['fsd'])
	with open(filename_sw, 'r') as f:
	       d = json.load(f)
	extract_dir = str(d['savedir'])
	acq = d['Acquisition']
	epochlen = int(d['epochlen'])

	fmax = 24
	fmin = 1

	for a in acq:
		EEG = np.load(os.path.join(extract_dir, 'downsampEEG_Acq'+str(a)+'.npy'))
		if int(d['emg']) == 1:
			EMG = np.load(os.path.join(extract_dir, 'downsampEMG_Acq'+str(a)+'.npy'))
			assert np.size(EEG) == np.size(EMG)

		acq_len = np.size(EEG)/fsd
		hour_segs = math.ceil(acq_len/3600)

		for h in np.arange(hour_segs):
			if hour_segs == 1:
				this_eeg = EEG
				if int(d['emg']) == 1:
					this_emg = EMG
			elif h == hour_segs-1:
				this_eeg = EEG[h*3600*fsd:]
				if int(d['emg']) == 1:
					this_emg = EMG[h*3600*fsd:]
			else:
				this_eeg = EEG[h*3600*fsd:(h+1)*3600*fsd]
				if int(d['emg']) == 1:
					this_emg = EMG[h*3600*fsd:(h+1)*3600*fsd]
			seg_len = np.size(this_eeg)/fsd
			nearest_epoch = math.floor(seg_len/epochlen)
			new_length = int(nearest_epoch*epochlen*fsd)
			this_eeg = this_eeg[0:new_length]
			t_vect = np.linspace(0,nearest_epoch*epochlen, np.size(this_eeg))
			freq, t_spec, x_spec = signal.spectrogram(this_eeg, fs=fsd, window='hanning', 
				nperseg=1000, noverlap=1000-1, mode='psd')
			
			delt = sum(x_spec[np.where(np.logical_and(freq>=1,freq<=4))])
			thetw = sum(x_spec[np.where(np.logical_and(freq>=2,freq<=16))])
			thetn = sum(x_spec[np.where(np.logical_and(freq>=5,freq<=10))])
			thet = thetn/thetw
			delt = (delt-np.average(delt))/np.std(delt)
			thet = (thet-np.average(thet))/np.std(thet)
			delt_time = np.vstack((delt, t_spec))
			thet_time = np.vstack((thet, t_spec))
			np.save(os.path.join(extract_dir,'delt' + str(a) + '_hr' + str(h)+ '.npy'),delt_time)
			np.save(os.path.join(extract_dir,'thet' + str(a) + '_hr' + str(h)+ '.npy'),thet_time)
			np.save(os.path.join(extract_dir, 'downsampEEG_Acq'+str(a) + '_hr' + str(h)+ '.npy'), this_eeg)
			np.save(os.path.join(extract_dir, 'downsampEMG_Acq'+str(a) + '_hr' + str(h)+ '.npy'), this_emg)

			del(x_spec)
			del(freq)
			del(t_spec)
			del(this_eeg)
			del(this_emg)
			

















