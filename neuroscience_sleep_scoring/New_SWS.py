import numpy as np
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import glob
from copy import deepcopy
import sys
import os
import math
import json
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import warnings
from neuroscience_sleep_scoring import SWS_utils, extract_data
from datetime import datetime
from neuroscience_sleep_scoring.SW_Cursor import Cursor
from neuroscience_sleep_scoring.SW_Cursor import ScoringCursor
import pathlib
import time
from datetime import datetime, timedelta
from scipy import io

key_stroke = 0

def on_press(event):
	global key_stroke
	if event.key in ['1','2','3', '4']:
		key_stroke = int(event.key)
		print(f'scored: {event.key}')
	elif event.key == 'q':
		print('QUIT')
		plt.close('all')
		sys.exit()
	else:
		key_stroke = np.float('nan')
		print('I did not understand that keystroke; I will mark it white and please come back to fix it.')

def update_model(d, FeatureDict):
	# Feed the data to retrain a model.
	if d['movement']:
		FeatureDict = SWS_utils.adjust_movement(FeatureDict, epochlen = d['epochlen'])

	FeatureDict = SWS_utils.adjust_movement(FeatureDict, d['movement'], epochlen = d['epochlen'])

	if 'EMGvar' in FeatureDict.keys():
		FeatureDict['EMGvar'][np.isnan(FeatureDict['EMGvar'])] = 0
	df_additions = pd.DataFrame(FeatureDict)
	# df_additions[pd.isnull(FeatureDict['EMGvar'])] = 0
	mod_name = d['mod_name']
	if len(d['EEG channel']) == 2:
		mod_name = mod_name+'_2chan'
	Sleep_Model = SWS_utils.update_sleep_df(d['model_dir'], mod_name, df_additions)
	jobname = SWS_utils.build_joblib_name(d)
	x_features = SWS_utils.get_xfeatures(FeatureDict)
	if 'EMGvar' in Sleep_Model.columns:
		Sleep_Model = Sleep_Model.drop(index=np.where(Sleep_Model['EMGvar'].isin(['nan']))[0])
	SWS_utils.retrain_model(Sleep_Model, x_features, d['model_dir'], jobname)


def display_and_fix_scoring(d, a, h, this_emg, State_input, is_predicted, clf, Features, this_video,
	acq_start, v = None, movement_df = None, buffer = 4):
	plt.ion()
	i = 0
	this_bin = 1*d['fsd']*d['epochlen'] #number of EEG data points in one epoch
	eeg_AD0 = np.load(os.path.join(d['savedir'],'AD0_downsampled', 
		'downsampEEG_Acq'+a+'_hr'+str(h)+'.npy'))
	eeg_AD2 = np.load(os.path.join(d['savedir'],'AD2_downsampled', 
		'downsampEEG_Acq'+a+'_hr'+str(h)+'.npy'))

	EEG_t = np.arange(np.size(eeg_AD0))/d['fsd'] #time array for EEG data
	start_trace = int(i-(4*d['epochlen'])) #timepoint in seconds that the plotted trace will start
	end_trace = int(i + (5*d['epochlen'])) #timepoint in seconds that the plotted trace will end

	if d['vid']:
		timestamp_df = pd.read_pickle(os.path.join(d['savedir'], 'All_timestamps.pkl'))
		try:
			this_timestamp = SWS_utils.pulling_timestamp(timestamp_df, acq_start, eeg_AD0, d['fsd'])
			cap, fps = SWS_utils.load_video(d, this_timestamp)
		except IndexError:
			d['vid'] = 0
			print("Timestamp information not available, turning off video access for this acquisition")

	print('loading the theta ratio...')
	ThD = SWS_utils.get_ThD(eeg_AD2, d['fsd']) #array of ThD values per second
	ThD_t = np.arange(0, np.size(ThD))

	fig2, (ax6, ax7, ax8, ax9, ax10) = plt.subplots(nrows=5, ncols=1, figsize=(14, 7.5))
	fig1, ax1, ax2, ax3, ax4, ax5 = SWS_utils.create_prediction_figure(d, State_input, is_predicted, clf, 
		Features, d['fsd'], eeg_AD0, eeg_AD2, this_emg, EEG_t, d['epochlen'], start_trace, end_trace, 
		d['Maximum_Frequency'], d['Minimum_Frequency'], [ax6, ax7], v = v)
	
	v_ylims = list(ax4.get_ylim())
	emg_ylims = list(ax5.get_ylim())

	buffer_seconds = buffer*d['epochlen'] #amount of time in seconds added to beginning and end of trace to accomodate looking at early and late epochs
	long_ThD, long_ThD_t = SWS_utils.add_buffer(ThD, ThD_t, buffer_seconds, fs = 1)
	long_emg, long_emg_t = SWS_utils.add_buffer(this_emg, EEG_t, buffer_seconds, fs = 200)
	if d['movement']:
		long_v, long_v_t = SWS_utils.add_buffer(np.insert(v[0],0,0), np.insert(v[1],0,0), 
			buffer_seconds, fs = 1/int(d['epochlen']))
	else:
		long_v = None
		long_v_t = None

	line1, line2, line3 = SWS_utils.create_zoomed_fig(ax8, ax9, ax10, long_emg, long_emg_t, 
		long_ThD, long_ThD_t, long_v, long_v_t, start_trace, end_trace, 
		epochlen = d['epochlen'], ThD_ylims = [0,30], emg_ylims = ([-0.25, 0.25]), v_ylims = v_ylims)


	ax6.set_xlim([-600, 600])
	ax7.set_xlim([-600, 600])
	line4 = ax6.axvline(0, linewidth = 2, color = 'k')
	line5 = ax7.axvline(0, linewidth = 2, color = 'k')

	fig2.tight_layout()
	markers = SWS_utils.make_marker(fig1, this_bin/d['fsd'], d['epochlen'])


	plt.ion()	
	State = deepcopy(State_input)
	#init cursor and it's libraries from SW_Cursor.py
	cursor = Cursor(ax1, ax2, ax5)	

	cID = fig1.canvas.mpl_connect('button_press_event', cursor.on_click)


	cID4 = fig1.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
	cID4 = fig1.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)

	#Ok so I think that the quotes is the specific event to trigger and the second arg is the function to run when that happens?
	cID2 = fig1.canvas.mpl_connect('axes_enter_event', cursor.in_axes)
	cID3 = fig1.canvas.mpl_connect('key_press_event', cursor.on_press)



	#This is the loop that manages the interface
	plt.show()
	DONE = False
	while not DONE:
		plt.waitforbuttonpress()

		if cursor.replot:
			print("Replot of fig 1. called!")
			this_epoch_t = math.floor(cursor.replotx/d['epochlen'])*d['epochlen']
			replot_start = start_trace + this_epoch_t
			replot_end = end_trace + this_epoch_t
			print('Epoch Start Time = ' + str(this_epoch_t) + ' seconds')
			print('Start Trace = '+str(replot_start) + ' seconds')
			print('End Trace = ' + str(replot_end) + ' seconds')

			SWS_utils.update_raw_trace(fig1, fig2, line1, line2, line3, line4, line5, long_emg, 
				long_emg_t, long_ThD, long_ThD_t, long_v, long_v_t, markers, this_epoch_t, 
				replot_start, replot_end, d['epochlen'])
			if d['vid']:
				if this_epoch_t-d['epochlen'] < 0:
					print('No video available for this bin')
				else:
					vid_start = int(this_timestamp.index[this_timestamp['Offset_Time']>(this_epoch_t-d['epochlen'])][0])
					vid_end = int(this_timestamp.index[this_timestamp['Offset_Time']<((this_epoch_t)+(d['epochlen']*2))][-1])
					this_timestamp['Offset_Time'][vid_start]

					SWS_utils.pull_up_movie(d, cap, vid_start, vid_end, 
						this_video, d['epochlen'], this_timestamp)


			plt.show()
			cursor.replot = False


			# Flip back the params

		if cursor.change_bins:
			bins = np.sort(cursor.bins)
			start_bin = cursor.bins[0]
			end_bin = cursor.bins[1]
			print(f'changing bins: {start_bin} to {end_bin}')
			SWS_utils.clear_bins(bins, ax2)
			fig2.canvas.draw()
			# new_state = int(input('What state should these be?: '))
			try:
				new_state = int(input('What state should these be?: '))
			except:
				new_state = int(input('What state should these be?: '))
			SWS_utils.correct_bins(start_bin, end_bin, ax2, new_state)
			fig2.canvas.draw()
			State[start_bin:end_bin] = new_state
			if end_bin == len(State)-1:
				State[end_bin] = new_state
			np.save(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)
			cursor.bins = []
			cursor.change_bins = False
		if cursor.DONE:
			DONE = True

	print('successfully left GUI')
	cv2.destroyAllWindows()
	plt.close('all')
	np.save(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)

	return State


def start_swscoring(d):
	# mostly for deprecated packages
	print('this code is supressing warnings')
	warnings.filterwarnings("ignore")
	print('These are the available acquisitions: '+ str(d['Acquisition']))
	state_files = glob.glob(os.path.join(d['savedir'], 'StatesAcq*.npy'))
	scored_acqs = []
	for sf in state_files:
		filename = os.path.split(sf)[1]
		idx1 = filename.find('q')
		idx2 = filename.find('_')
		try:
			acq_num = int(filename[idx1+1:idx2])
		except ValueError:
			continue
		scored_acqs.append(acq_num)
	print('These are the acquisitions that have a previous State file: ' + str(sorted(scored_acqs)))
	a = input('Which acqusition do you want to score?')

	print('Loading EEG and EMG....')
	downsampEEG = np.load(os.path.join(d['savedir'],'downsampEEG_Acq'+str(a)+'.npy'))
	downsampEMG = np.load(os.path.join(d['savedir'],'downsampEMG_Acq'+str(a)+'.npy'))

	acq_len = np.size(downsampEEG)/d['fsd'] # fs: sampling rate, fsd: downsampled sampling rate
	hour_segs = math.ceil(acq_len/3600) # acq_len in seconds, convert to hours
	print('This acquisition has ' +str(hour_segs)+ ' segments.')

	acq_start = SWS_utils.get_AcqStart(d, a, acq_len)

	for h in np.arange(hour_segs):
		# FeatureDict = {}
		eeg_df = pd.DataFrame()
		normVal = []
		for e in d['EEG channel']:
			eeg_dir = os.path.join(d['savedir'], 'AD'+str(e)+'_downsampled')
			eeg_df['EEGChannel'+str(e)] = np.load(os.path.join(eeg_dir, 'downsampEEG_Acq'+a+'_hr'+str(0)+'.npy'))
			normVal.append(np.load(os.path.join(eeg_dir, d['basename']+'_normVal.npy')))

		eeg_df['EMG'] = np.load(os.path.join(d['savedir'],'downsampEMG_Acq'+str(a) + '_hr' + str(h)+ '.npy'))
	
		# chop off the remainder that does not fit into the 4s epoch
		seg_len = len(eeg_df)/d['fsd']
		nearest_epoch = math.floor(seg_len/d['epochlen'])
		new_length = int(nearest_epoch*d['epochlen']*d['fsd'])
		eeg_df = eeg_df.iloc[:new_length]
		FeatureDict = SWS_utils.build_feature_dict(eeg_df, d['fsd'], d['epochlen'],
			normVal = normVal)
		this_video, v, this_motion = SWS_utils.initialize_vid_and_move(d, a, acq_start, acq_len)
		if d['movement']:
			FeatureDict['Velocity'] = v[0]
		FeatureDict['animal_name'] = np.full(len(FeatureDict[list(FeatureDict.keys())[0]]), d['mouse_name'])

		os.chdir(d['savedir'])

		check = input('Do you want to check and fix existing scoring (c) or score new dataset (s)?: c/s ')
		while check != 'c' and check != 's':
			check = input(
				'Only c/s is accepted. Do you want to check and fix existing scoring (c) or score new dataset (s)?: c/s ')
		if check == 'c':
			try:
				# if some portion of the file has been previously scored
				State = np.load(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'))
				wrong, = np.where(np.isnan(State))
				State[wrong] = 0
				s, = np.where(State == 0)
				this_emg = eeg_df['EMG']
				State = display_and_fix_scoring(d, a, h, this_emg, State, False, None,
										None, this_video, acq_start, v = v, movement_df = this_motion)
				if np.any(State == 0):
					print('The following bins are not scored: \n' + str(np.where(State == 0)[0])  )
					zero_check = input('Do you want to go back and fix this right now? (y/n)' ) == 'y'
					if zero_check:
						State = display_and_fix_scoring(d, a, h, this_emg, State, False, None,
										None, this_video, acq_start, v = v, movement_df = this_motion)					
					else:
						print('Ok, but please do not update the model until you fix them')
			except FileNotFoundError:
				# if the file is a brand new one for scoring
				print("There is no existing scoring.")

		elif check == 's':
			model = input('Use a random forest? y/n: ') == 'y'

			if model:
				jobname = SWS_utils.build_joblib_name(d)
				try:
					clf = joblib.load(os.path.join(d['model_dir'], jobname))
				except FileNotFoundError:
					print("You don't have a model to work with.")
					return

				# feature list
				this_emg = eeg_df['EMG']

				Features = SWS_utils.prepare_feature_data(FeatureDict, d['movement'])

				Predict_y = clf.predict(Features)
				Predict_y = SWS_utils.fix_states(Predict_y)
				np.save(os.path.join(d['savedir'], 'model_prediction_Acq' + str(a) + '_hr' + str(h) + '.npy'), Predict_y)
				State = display_and_fix_scoring(d, a, h, this_emg, Predict_y, True, clf,
					Features, this_video, acq_start, v = v, movement_df = this_motion)
			else:
				State = np.zeros(int(acq_len/d['epochlen']))
				State = display_and_fix_scoring(d, a, h, this_emg, State, False, None,
										None, this_video, acq_start, v = v, movement_df = this_motion)
		
		FeatureDict['State'] = State

		update = input('Do you want to update the model?: y/n ') == 'y'
		if update:
			update_model(d, FeatureDict)					
			model_log(d['modellog_dir'], 0, d['species'], d['mouse_name'], d['mod_name'], a)
		logq = input('Do you want to update your personal log?: y/n ') == 'y'
		if logq:
			personal_log(d['personallog_dir'], d['mouse_name'], d['savedir'], a)
			
		plt.close('all')
			# Store the result.

def load_data_for_sw(filename_sw, return_data = False):
	with open(filename_sw, 'r') as f:
		d = json.load(f)
	if return_data:
		return d
	start_swscoring(d)

def build_model(filename_sw):
	with open(filename_sw, 'r') as f:
		d = json.load(f)

	print('this code is supressing warnings')
	warnings.filterwarnings("ignore")
	extract_data.pulling_acqs(filename_sw)
	print('These are the available acquisitions: '+ str(d['Acquisition']))
	these_acqs = input('Which acqusitions do you want to use in the model?').split(',')
	eeg_dir = os.path.join(d['savedir'], 'AD' + str(d['EEG channel']) + '_downsampled')
	for a in these_acqs:
		print('Loading EEG and EMG....')
		downsampEEG = np.load(os.path.join(d['savedir'],'downsampEEG_Acq'+str(a)+'.npy'))
		if d['emg']:
			downsampEMG = np.load(os.path.join(d['savedir'],'downsampEMG_Acq'+str(a)+'.npy'))
		acq_len = np.size(downsampEEG)/d['fsd'] # fs: sampling rate, fsd: downsampled sampling rate
		acq_start = SWS_utils.get_AcqStart(d, a, acq_len)
		this_eeg = np.load(os.path.join(eeg_dir, 'downsampEEG_Acq'+str(a) + '_hr' + str(h)+ '.npy'))
		if d['emg']:
			this_emg = np.load(os.path.join(eeg_dir,'downsampEMG_Acq'+str(a) + '_hr' + str(h)+ '.npy'))
		else:
			this_emg = None
		# chop off the remainder that does not fit into the 4s epoch
		seg_len = np.size(this_eeg)/d['fsd']
		nearest_epoch = math.floor(seg_len/d['epochlen'])
		new_length = int(nearest_epoch*d['epochlen']*d['fsd'])
		this_eeg = this_eeg[0:new_length]
		normVal = np.load(os.path.join('/Volumes/yaochen/Active/Lizzie/FLP_data/',d['basename'],d['basename']+'_extracted_data/',d['basename']+'_normVal.npy'))

		FeatureDict = SWS_utils.build_feature_dict(this_eeg, d['fsd'], d['epochlen'], 
			this_emg = this_emg, normVal =normVal)
		this_video, v, this_motion = SWS_utils.initialize_vid_and_move(d, a, acq_start, acq_len)
		FeatureDict['Velocity'] = v[0]
		FeatureDict['animal_name'] = np.full(np.size(FeatureDict['delta_pre']), d['mouse_name'])
		try:
			State = np.load(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr0.npy'))
			wrong, = np.where(np.isnan(State))
			State[wrong] = 0
			State = display_and_fix_scoring(d, a, 0, this_emg, State, False, None,
									None, this_video, acq_start, v = v, movement_df = this_motion)
			FeatureDict['State'] = State
			keep = input('Do you want this to be part of the model? (y/n)') == 'y'
			if keep:
				update_model(d, FeatureDict)
				model_log(d['modellog_dir'], 2, d['species'], d['mouse_name'], d['mod_name'], a)
			else:
				continue

		except FileNotFoundError:
			# if the file is a brand new one for scoring
			print("There is no existing scoring.")

def model_log(log_dir, action, animal, mouse_name, mod_name, a):
	log_file = os.path.join(log_dir, mod_name+'_scoringlog.txt')
	if not os.path.exists(log_file):
		print(log_file + ' does not exist. Making it now')
		f = open(log_file, "w+")
		f.close()

	state_dict = { '0': 'corrected',
					'1': 'scored with ML model',
					'2': 'scored in legacy mode'
	}

	print("Logging to " + log_file)

	file = open(log_file, "a+")

	# datetime object containing current date and time
	now = datetime.now()

	# dd/mm/YY H:M:S
	dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
	
	whois = input("What is your name?:")
	file.write(animal + " " + mouse_name + " acquisition " + str(a) +  " was " + 
		state_dict[str(action)]  + " by " + whois + " on " + dt_string + "\n")
	file.flush()
	file.close()
def personal_log(log_dir, mouse_name, save_dir, a):
	log_file = os.path.join(log_dir,'personal_scoringlog.csv')
	if not os.path.exists(log_file):
		print(log_file + ' does not exist. Making it now')
		df = pd.DataFrame(columns=['Date', 'Mouse Name', 'Acquisition', 'State Array Location'])
		df.to_csv(log_file, mode='a', header=True, index=False)
	d = {'Date': [pd.Timestamp.now()], 'Mouse Name': [mouse_name], 'Acquisition': [a],'State Array Location': [save_dir]}
	df = pd.DataFrame(data=d)
	df.to_csv(log_file, mode='a', header=False, index=False)

if __name__ == "__main__":
	args = sys.argv
	# Why do we need to assert this??? Why the heck would you care if you execute from the same dir if we don't use relative paths anywhere else in the code
	# assert args[0] == 'New_SWS.py'
	if len(args) < 2:
		print("You need to specify the path of your Score_Settings.json. For instance, run `python New_SWS.py /home/ChenLab_Sleep_Scoring/Score_Settings.json`.")
	elif len(args) > 2:
		print("You only need to specify the path of your Score_Settings.json. For instance, run `python New_SWS.py /home/ChenLab_Sleep_Scoring/Score_Settings.json`.")
	else:
		load_data_for_sw(args[1])
