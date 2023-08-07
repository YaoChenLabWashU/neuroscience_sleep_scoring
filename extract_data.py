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
import time
from datetime import datetime



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
		if int(d['emg']) == 1:
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
			if int(d['emg']) == 1:
				np.save(os.path.join(savedir, 'downsampEMG_Acq'+str(a) + '_hr' + str(h)+ '.npy'), this_emg)

def combine_bonsai_data(filename_sw, d):
	videos = glob.glob(os.path.join(d['video_dir'], '*.mp4'))
	if len(videos) == 0:
		videos = glob.glob(os.path.join(d['video_dir'], '*.avi'))
	if len(videos) == 0:
		print('No videos found! Please check directory')
		sys.exit()
	videos = SWS_utils.sort_files(videos, d['basename'])
	all_ts_df  = pd.DataFrame(columns = ['Timestamps', 'Filename'])
	if d['movement']:
		if d['DLC']:
			all_move_df = pd.DataFrame(columns = ['Timestamps', 'X','Y','Likelihood','Filename'])
		else:
			all_move_df = pd.DataFrame(columns = ['Timestamps', 'X','Y', 'Filename'])
	
	for i, a in enumerate(d['Acquisition']):
		timestamp_df = SWS_utils.timestamp_extracting(d, a)
		all_ts_df  = pd.concat([all_ts_df, timestamp_df])
		if d['movement']:
			movement_df = SWS_utils.movement_extracting(d, a)
			bad_frames, = np.where(movement_df['Likelihood'] < 0.8)
			perc_bad = np.size(bad_frames)/len(movement_df.index)
			if perc_bad > 0.15:
				new_label = alternate_label(this_video, csv_dir, i)
				movement_df = SWS_utils.movement_extracting(csv_dir, d['Acquisition'], a, bonsai_v, 
				d['DLC'], new_label, this_video = this_video)
			
			movement_df['Timestamps'] = timestamp_df['Timestamps']
			all_move_df  = pd.concat([all_move_df, movement_df])
	all_ts_df.to_pickle(os.path.join(d['savedir'], 'All_timestamps.pkl'))
	if d['movement']:
		all_move_df.to_pickle(os.path.join(d['savedir'], 'All_movement.pkl'))
def pulling_acqs(filename_sw):
	with open(filename_sw, 'r') as f:
			d = json.load(f)
	AD_file = glob.glob(os.path.join(d['rawdat_dir'], 'AD0_*'))
	acqs = []
	for fn in AD_file:
		filename = os.path.split(fn)[1]
		idx1 = filename.find('_')
		idx2 = filename.find('.mat')
		try:
			acq_num = int(filename[idx1+1:idx2])
		except ValueError:
			continue
		acqs.append(acq_num)
	d['Acquisition'] = sorted(acqs)
	with open(filename_sw, 'w') as f:
		json.dump(d, f, indent=2)

def alternate_label(this_video, csv_dir, i):
	this_dir,fn = os.path.split(this_video)
	labeled_fn = glob.glob(os.path.join(os.path.split(this_dir)[0], 
		'DLC_outputs', fn[:fn.find('.')] + '*labeled.mp4'))[0]
	print('Pull up this video file: '+ labeled_fn)
	csv_file = glob.glob(os.path.join(csv_dir, '*_motion'+ str(i) + '.csv'))[0]
	SWS_utils.DLC_check_fig(csv_file)
	new_label = input('What label do you want to use?')
	return new_label

def make_full_velocity_array(savedir):
	movement_df = pd.read_pickle(os.path.join(savedir, 'All_movement.pkl'))
	v = SWS_utils.movement_processing(movement_df)
	np.save(os.path.join(savedir, 'velocity_vector.npy'), v)

def get_normalizing_value(filename_sw):
	with open(filename_sw, 'r') as f:
		d = json.load(f)
	eeg_files = glob.glob(os.path.join(d['savedir'], 'downsampEEG_Acq*_hr0.npy'))
	all_tp = []
	for f in eeg_files:
		this_eeg = np.load(f)
		all_tp.append(SWS_utils.get_total_power(this_eeg, d['fsd']))
	normVal = np.median(np.concatenate(all_tp))
	np.save(os.path.join(d['savedir'], d['basename']+'_normVal.npy'), normVal)

	return
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
		choosing_acquisition(args[1])
		downsample_filter(args[1])
		get_normalizing_value(args[1])
		if d['movement']:
			combine_bonsai_data(args[1], d)
			plt.close('all')
			velocity_curve = input('Do you want to make the full velocity array (y/n)?')
			if velocity_curve == 'y':
				make_full_velocity_array(d['savedir'])
















