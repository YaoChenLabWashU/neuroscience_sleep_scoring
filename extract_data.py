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
import json
import sys
from neuroscience_sleep_scoring import SWS_utils
import pandas as pd


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
	print('Here are the acquisitions you chose: ')
	print(sorted(acq))

	with open(filename_sw, 'r') as f:
		d = json.load(f)

	d['Acquisition'] = sorted(acq)

	with open(filename_sw, 'w') as f:
		json.dump(d, f, indent=2)

def downsample_filter(filename_sw):
	with open(filename_sw, 'r') as f:
			d = json.load(f)

	rawdat_dir = str(d['rawdat_dir'])
	model_dir = str(d['model_dir'])
	animal = str(d['species'])
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
		a = acq[fil]
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
			#emg_abs = np.absolute(emg_downsamp)
		eeg_downsamp = signal.resample(eegfilt, int(new_len))
		np.save(os.path.join(savedir, 'downsampEEG_Acq'+str(a)), eeg_downsamp)
		np.save(os.path.join(savedir, 'downsampEMG_Acq'+str(a)), emg_downsamp)

		acq_len = np.size(eeg_downsamp)/fsd
		hour_segs = math.ceil(acq_len/3600)

		for h in np.arange(hour_segs):
			if hour_segs == 1:
				this_eeg = eeg_downsamp
				if int(d['emg']) == 1:
					this_emg = emg_downsamp
			elif h == hour_segs-1:
				this_eeg = eeg_downsamp[h*3600*fsd:]
				if int(d['emg']) == 1:
					this_emg = emg_downsamp[h*3600*fsd:]
			else:
				this_eeg = eeg_downsamp[h*3600*fsd:(h+1)*3600*fsd]
				if int(d['emg']) == 1:
					this_emg = emg_downsamp[h*3600*fsd:(h+1)*3600*fsd]
			seg_len = np.size(this_eeg)/fsd
			nearest_epoch = math.floor(seg_len/epochlen)
			new_length = int(nearest_epoch*epochlen*fsd)
			this_eeg = this_eeg[0:new_length]			
			np.save(os.path.join(savedir, 'downsampEEG_Acq'+str(a) + '_hr' + str(h)+ '.npy'), this_eeg)
			np.save(os.path.join(savedir, 'downsampEMG_Acq'+str(a) + '_hr' + str(h)+ '.npy'), this_emg)

def combine_bonsai_data(filename_sw):
	with open(filename_sw, 'r') as f:
			d = json.load(f)
	vid = d['vid']
	video_dir = d['video_dir']
	bonsai_v = d['Bonsai Version']
	acq = d['Acquisition']
	savedir = d['savedir']

	if bonsai_v < 6:
		csv_dir = video_dir

	if bonsai_v >= 6:
		if video_dir[-1] != '/':
		    video_dir = video_dir + '/'
		top_directory = video_dir.replace(video_dir.split('/')[-2], '')[:-1]
		csv_dir = glob.glob(top_directory + '*csv')[0]

	videos = glob.glob(os.path.join(video_dir, '*.mp4'))
	videos.sort(key=lambda f: os.path.getmtime(os.path.join(video_dir, f)))
	all_ts_df  = pd.DataFrame(columns = ['Timestamps', 'Filename'])
	all_move_df = pd.DataFrame(columns = ['Timestamps', 'X','Y', 'Filename'])
	
	for i, a in enumerate(acq):
		this_video = videos[i]
		cap, timestamp_df, fps = SWS_utils.load_video(this_video, bonsai_v, a, acq)
		movement_df = SWS_utils.movement_extracting(csv_dir, acq, a, bonsai_v, this_video = this_video)
		del cap
		all_ts_df  = all_ts_df.append(timestamp_df)
		all_move_df  = all_move_df.append(movement_df)
	all_ts_df.to_csv(os.path.join(savedir, 'All_timestamps.csv'))
	all_move_df.to_csv(os.path.join(savedir, 'All_movement.csv'))

	








# def create_spectrogram(filename_sw):
# 	with open(filename_sw, 'r') as f:
# 	       d = json.load(f)
# 	extract_dir = str(d['savedir'])
# 	acq = d['Acquisition']
# 	epochlen = int(d['epochlen'])
# 	fsd = int(d['fsd'])

# 	fmax = 24
# 	fmin = 1

# 	for a in acq:
# 		EEG = np.load(os.path.join(extract_dir, 'downsampEEG_Acq'+str(a)+'.npy'))
# 		if int(d['emg']) == 1:
# 			EMG = np.load(os.path.join(extract_dir, 'downsampEMG_Acq'+str(a)+'.npy'))
# 			assert np.size(EEG) == np.size(EMG)

# 		acq_len = np.size(EEG)/fsd
# 		hour_segs = math.ceil(acq_len/3600)

# 		for h in np.arange(hour_segs):
# 			if hour_segs == 1:
# 				this_eeg = EEG
# 				if int(d['emg']) == 1:
# 					this_emg = EMG
# 			elif h == hour_segs-1:
# 				this_eeg = EEG[h*3600*fsd:]
# 				if int(d['emg']) == 1:
# 					this_emg = EMG[h*3600*fsd:]
# 			else:
# 				this_eeg = EEG[h*3600*fsd:(h+1)*3600*fsd]
# 				if int(d['emg']) == 1:
# 					this_emg = EMG[h*3600*fsd:(h+1)*3600*fsd]
# 			seg_len = np.size(this_eeg)/fsd
# 			nearest_epoch = math.floor(seg_len/epochlen)
# 			new_length = int(nearest_epoch*epochlen*fsd)
# 			this_eeg = this_eeg[0:new_length]
# 			t_vect = np.linspace(0,nearest_epoch*epochlen, np.size(this_eeg))
# 			freq, t_spec, x_spec = signal.spectrogram(this_eeg, fs=fsd, window='hanning', 
# 				nperseg=1000, noverlap=1000-1, mode='psd')
			
# 			delt = sum(x_spec[np.where(np.logical_and(freq>=1,freq<=4))])
# 			thetw = sum(x_spec[np.where(np.logical_and(freq>=2,freq<=16))])
# 			thetn = sum(x_spec[np.where(np.logical_and(freq>=5,freq<=10))])
# 			thet = thetn/thetw
# 			delt = (delt-np.average(delt))/np.std(delt)
# 			thet = (thet-np.average(thet))/np.std(thet)
# 			delt_time = np.vstack((delt, t_spec))
# 			thet_time = np.vstack((thet, t_spec))
# 			np.save(os.path.join(extract_dir,'delt' + str(a) + '_hr' + str(h)+ '.npy'),delt_time)
# 			np.save(os.path.join(extract_dir,'thet' + str(a) + '_hr' + str(h)+ '.npy'),thet_time)
# 			np.save(os.path.join(extract_dir, 'downsampEEG_Acq'+str(a) + '_hr' + str(h)+ '.npy'), this_eeg)
# 			np.save(os.path.join(extract_dir, 'downsampEMG_Acq'+str(a) + '_hr' + str(h)+ '.npy'), this_emg)

# 			del(x_spec)
# 			del(freq)
# 			del(t_spec)
# 			del(this_eeg)
# 			del(this_emg)
	
if __name__ == "__main__":
	args = sys.argv
	# Why do we need to assert this??? Why the heck would you care if you execute from the same dir if we don't use relative paths anywhere else in the code
	# assert args[0] == 'New_SWS.py'
	if len(args) < 2:
		print("You need to specify the path of your Score_Settings.json. For instance, run `python New_SWS.py /home/ChenLab_Sleep_Scoring/Score_Settings.json`.")
	elif len(args) > 2:
		print("You only need to specify the path of your Score_Settings.json. For instance, run `python New_SWS.py /home/ChenLab_Sleep_Scoring/Score_Settings.json`.")
	else:
		with open(args[1], 'r') as f:
				d = json.load(f)
		movement = d['movement']
		choosing_acquisition(args[1])
		downsample_filter(args[1])
		if movement:
			combine_bonsai_data(args[1])
















