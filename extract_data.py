from tabnanny import check
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
import PKA_Sleep as PKA
import natsort, pyedflib
from scipy import signal



def choosing_acquisition(filename_sw):
	with open(filename_sw, 'r') as f:
		d = json.load(f)

	rawdat_dir = str(d['rawdat_dir'])
	fs = int(d['fs'])
	EEG_chan = d['EEG channel']

	print(rawdat_dir)
	os.chdir(rawdat_dir)

	poss_files = glob.glob('AD'+str(EEG_chan[0])+'*.mat')
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

def downsample_filter(filename_sw, EEG_channels = ['0','2']):
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
	os.makedirs(savedir, exist_ok = True)

	print(rawdat_dir)
	os.chdir(rawdat_dir)
	nyq = 0.5*fs
	N  = 3    # Filter order

	for EEG_chan in EEG_channels:
		os.makedirs(os.path.join(savedir, 'AD'+EEG_chan+'_downsampled'), exist_ok = True)
		# if emg_flag:
		# 	os.makedirs(os.path.join(savedir, 'AD'+EMG_chan+'_downsampled'), exist_ok = True)
		EEG_files = [glob.glob('AD'+EEG_chan+'_'+str(i)+'.mat') for i in acq]
		EEG_files = np.asarray(np.concatenate(EEG_files))

		EMG_files = [glob.glob('AD'+EMG_chan+'_'+str(i)+'.mat') for i in acq]
		EMG_files = np.asarray(np.concatenate(EMG_files))

		

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
				np.save(os.path.join(savedir, 'AD'+str(EEG_chan)+'_downsampled', 'downsampEEG_Acq'+str(a) + '_hr' + str(h)+ '.npy'), this_eeg)
				if int(d['emg']) == 1:
					np.save(os.path.join(savedir, 'AD'+str(EEG_chan)+'_downsampled', 'downsampEMG_Acq'+str(a) + '_hr' + str(h)+ '.npy'), this_emg)

def combine_bonsai_data(filename_sw, d):
	videos = glob.glob(os.path.join(d['video_dir'], '*.mp4'))
	if len(videos) == 0:
		videos = glob.glob(os.path.join(d['video_dir'], '*.avi'))
	if len(videos) == 0:
		print('No videos found! Please check directory')
		sys.exit()
	videos = SWS_utils.sort_files(videos, d['basename'], d['csv_dir'])

	timestamp_files = glob.glob(os.path.join(d['csv_dir'], '*imestamp*.csv'))
	timestamp_files = SWS_utils.sort_files(timestamp_files, d['basename'], d['csv_dir'])
	all_ts_df  = pd.DataFrame(columns = ['Timestamps', 'Filename'])
	
	if d['movement']:
		if d['DLC']:
			all_move_df = pd.DataFrame(columns = ['Timestamps', 'X','Y','Likelihood','Filename'])
		else:
			all_move_df = pd.DataFrame(columns = ['Timestamps', 'X','Y', 'Filename'])
		movement_files = glob.glob(os.path.join(d['csv_dir'], '*motion*.csv'))
		movement_files = SWS_utils.sort_files(movement_files, d['basename'], d['csv_dir'])
		if len(timestamp_files) != len(movement_files):
			print('There is a different number of timestamp files and movement files. This will cause misalignment. Please Check this.')
			print(timestamp_files, sep="\n")
			print(movement_files, sep="\n")
			sys.exit()

	for i in range(len(timestamp_files)):
		timestamp_df = SWS_utils.timestamp_extracting(timestamp_files[i])
		all_ts_df  = pd.concat([all_ts_df, timestamp_df])
		if d['movement']:
			movement_df = SWS_utils.movement_extracting(movement_files[i], d)
			bad_frames, = np.where(movement_df['Likelihood'] < 0.8)
			perc_bad = np.size(bad_frames)/len(movement_df.index)			
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

def make_full_velocity_array(savedir, binsize = 4, return_array = False):
	movement_df = pd.read_pickle(os.path.join(savedir, 'All_movement.pkl'))
	v = SWS_utils.movement_processing(movement_df, binsize = binsize)
	np.save(os.path.join(savedir, 'velocity_vector.npy'), v)
	if return_array:
		return v

def get_normalizing_value(filename_sw, EEG_channels = ['0','2']):
	with open(filename_sw, 'r') as f:
		d = json.load(f)
	for EEG_chan in EEG_channels:
		eeg_files = glob.glob(os.path.join(d['savedir'], 'AD'+str(EEG_chan)+'_downsampled','downsampEEG_Acq*_hr0.npy'))
		all_tp = []
		for f in eeg_files:
			this_eeg = np.load(f)
			all_tp.append(SWS_utils.get_total_power(this_eeg, d['fsd']))
		normVal = np.median(np.concatenate(all_tp))
		np.save(os.path.join(d['savedir'], 'AD'+str(EEG_chan)+'_downsampled', d['basename']+'_normVal.npy'), normVal)

def full_EEG_EMG(d, EEG_channels = ['0','2']):
	for EEG_chan in EEG_channels:
		full_EEG = PKA.get_all_EEG(os.path.join(d['savedir'], 'AD'+str(EEG_chan)+'_downsampled'), concatenate = True)
		np.save(os.path.join(d['savedir'], 'AD'+str(EEG_chan)+'_downsampled', 'AD'+ EEG_chan + '_full.npy'), full_EEG)
	if int(d['emg']) == 1:
		full_EMG = PKA.get_all_EEG(d['savedir'], concatenate = True, EMG_flag = True)
		np.save(os.path.join(d['savedir'], 'AD3_full.npy'), full_EMG)
  
def save_to_edf(data, filename, sample_rate,channel_labels):
	"""
	Save a NumPy array (EEG/EMG data) to an EDF file.
		
	Parameters:
	- data: 2D NumPy array of shape (channels, samples)
	- filename: Name of the EDF file to save
	- sample_rate: Sampling rate in Hz
	"""
	num_channels, num_samples = data.shape
	physical_min, physical_max = data.min(), data.max()
	digital_min, digital_max = -32768, 32767  # Standard EDF digital range
	signal_headers = []
	for label in channel_labels:
		signal_headers.append({
			"label": label,
			"dimension": "uV",
			"sample_frequency": sample_rate,
			"physical_min": physical_min,
			"physical_max": physical_max,
			"digital_min": digital_min,
			"digital_max": digital_max
		})
	with pyedflib.EdfWriter(filename, num_channels, file_type=pyedflib.FILETYPE_EDF) as f:
		f.setSignalHeaders(signal_headers)
		f.setStartdatetime(datetime.now())  # Set current timestamp
		f.writeSamples(data)
		f.close()
	print(f"Saved EDF file: {filename}")
  
def make_edf_file(d,highpass_eeg = True, emg_highpass = 20,
				  new_fs=250,chunk_size_hours = 24,check_emg_artifacts=False):
	'''
	This function will take the EEG and EMG data and save it to an EDF file.
	d can also be a dictionary:
	e.g.: d = {
		'basename': 'mouse1',
		'rawdat_dir': '/path/to/rawdata',
		'fs': 2000}
	'''
	def make_numpy_files(fileglob,savename,highpass):
		files = glob.glob(fileglob)
		files = [f for f in natsort.natsorted(files) if 'avg' not in f]
		
		print('loading files')
		eeg = np.concatenate([scipy.io.loadmat(f)[os.path.split(f)[1][:-4]][0][0][0][0] for f in files])
		fs = d['fs']
		nyq = 0.5*fs
		if highpass_eeg:
			b,a = signal.butter(3, highpass,fs=fs, btype = 'highpass',output='ba')
			eeg = signal.filtfilt(b,a,eeg)
		print('saving numpy file: %s'%savename)
		np.save(savename,eeg)
		return eeg
		
	fs = d['fs']
	animal= d['basename']
	saved_edf_file_name = animal + f'_eegemg_%dHz_%dchunk_%d.edf'
	datadir = d['rawdat_dir']
	savedir = os.path.join(datadir, animal+'_edffiles')
	eeg1_save = os.path.join(savedir,animal+'_AD0_full_highpass%s.npy'%highpass_eeg)
	eeg2_save = os.path.join(savedir,animal+'_AD2_full_highpass%s.npy'%highpass_eeg)
	emg_save = os.path.join(savedir,animal+'_AD3_full_highpass%d.npy'%emg_highpass)
	os.makedirs(savedir, exist_ok = True)
	#
	if not os.path.exists(eeg1_save):
		eeg1 = make_numpy_files(os.path.join(datadir,'AD0*.mat'),eeg1_save,highpass_eeg)
	else:
		eeg1 = np.load(eeg1_save)
	#	
	if not os.path.exists(eeg2_save):
		eeg2 = make_numpy_files(os.path.join(datadir,'AD2*.mat'),eeg2_save,highpass_eeg)
	else:
		eeg2 = np.load(eeg2_save)
	#	
	if not os.path.exists(emg_save):
		emg = make_numpy_files(os.path.join(datadir,'AD3*.mat'),emg_save,emg_highpass)
	else:
		emg = np.load(emg_save)
	if check_emg_artifacts:
		thresh = 5.5*np.std(emg) + np.mean(emg)
		clipped_indxs = emg > thresh
		emg[clipped_indxs] = thresh
		print('Clipped EMG at %d spots'%sum(clipped_indxs))
		
	eeg_emg_data = np.column_stack((eeg1,eeg2,emg)).squeeze().T
	del eeg1,eeg2,emg
	sample_rate = new_fs
	if new_fs != fs:
		up = int(new_fs/np.gcd(new_fs,fs))
		down = int(fs/np.gcd(new_fs,fs))
		eeg_emg_data = signal.resample_poly(eeg_emg_data,up,down,axis=1)
		
	days = int(eeg_emg_data.shape[1]/(sample_rate*3600*chunk_size_hours))

	for d in range(days):
		filename = saved_edf_file_name % (sample_rate,chunk_size_hours,d)
		print(filename)
		rec_end = sample_rate*3600*chunk_size_hours*(d+1) if sample_rate*3600*chunk_size_hours*(d+1) <= eeg_emg_data.shape[1] else eeg_emg_data.shape[1]
		save_to_edf(eeg_emg_data[:,sample_rate*3600*chunk_size_hours*d:rec_end], 
				savedir + os.sep + filename, 
				sample_rate, 
				['EEG_AD0', 'EEG_AD2', 'EMG_filt'])  # Save to EDF file


    
 
 
    
    

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
		make_edf_file(d,highpass_eeg = True, emg_highpass = 20,
                new_fs=250,chunk_size_hours = 24,check_emg_artifacts=True)
		if d['movement']:
			combine_bonsai_data(args[1], d)
			plt.close('all')
			velocity_curve = input('Do you want to make the full velocity array (y/n)?')
			if velocity_curve == 'y':
				make_full_velocity_array(d['savedir'])
