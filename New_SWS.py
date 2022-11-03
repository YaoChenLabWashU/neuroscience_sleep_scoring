import numpy as np
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import glob
import copy
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
from neuroscience_sleep_scoring import SWS_utils
from datetime import datetime
from neuroscience_sleep_scoring.SW_Cursor import Cursor
from neuroscience_sleep_scoring.SW_Cursor import ScoringCursor
import pathlib
import time
from datetime import datetime

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


def manual_scoring(d, a, this_eeg, this_emg, this_video, h, EEG_datetime, v = None):
	# Manually score the entire file.
	plt.ion()
	fig, (ax1, ax2, ax4, axx) = plt.subplots(nrows=4, ncols=1, figsize=(11, 6))
	fig2, ax5, ax6 = SWS_utils.create_scoring_figure(d['savedir'], a, this_eeg, d['fsd'], d['Maximum_Frequency'], 
		d['Minimum_Frequency'], movement_flag = d['movement'], v = v)
	
	cID2 = fig.canvas.mpl_connect('key_press_event', on_press)
	cID3 = fig2.canvas.mpl_connect('key_press_event', on_press)
	i = 0
	this_bin = 1*d['fsd']*d['epochlen']
	realtime = np.arange(np.size(this_eeg)) / d['fsd']

	start_trace = int(i * d['fsd'] * d['epochlen'])
	end_trace = int(start_trace + d['fsd'] * 11 * d['epochlen'])
	LFP_ylim = 5

	DTh = SWS_utils.load_bands(this_eeg, d['fsd'])
	if vid_flag:
		print('Loading video now, this might take a second....')
		cap, timestamp_df, fps = SWS_utils.load_video(this_video, a, d['Acquisition'], d['savedir'])
		timestamp_df = SWS_utils.pulling_timestamp(timestamp_df, EEG_datetime, this_eeg, d['fsd'])


	line1, line2, line4, line5 = SWS_utils.raw_scoring_trace(ax1, ax2, ax4, axx, d['emg'], start_trace, 
								end_trace, realtime, this_eeg, d['fsd'], LFP_ylim, DTh, 
								d['epochlen'], this_emg)

	marker1, marker2 = SWS_utils.make_marker(ax5, None, this_bin, realtime, 
						d['fsd'], d['epochlen'], num_markers = 1)

	fig.show()
	fig2.show()
	fig.tight_layout()
	fig2.tight_layout()

	plt.show()
	

	try:
		# if some portion of the file has been previously scored
		State = np.load(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'))
		wrong, = np.where(np.isnan(State))
		State[wrong] = 0
		s, = np.where(State == 0)
		color_dict = {'0': 'white',
					  '1': 'green',
					  '2': 'blue',
					  '3': 'red',
					  '4': 'purple'}
		# rendering what has been previously scored
		for count, color in enumerate(State[:-1]):
			start = int(count * d['fsd'] * d['epochlen'])
			rect = patch.Rectangle((realtime[start + (d['epochlen'] * d['fsd'])], 0),
								   (d['epochlen']), 1, color=color_dict[str(int(color))])
			ax6.add_patch(rect)
		fig2.show()

	except FileNotFoundError:
		# if the file is a brand new one for scoring
		State = np.zeros(int(np.size(this_eeg) / d['fsd'] / d['epochlen']))
		s = np.arange(1, np.size(State) - 1)
		first_state = int(input('Enter the first state: '))
		State[0] = first_state

	for i in s[:-3]:
		# input('press enter or quit')
		print(f'here. index: {i}')
		start_trace = int(i * d['fsd'] * d['epochlen'])
		end_trace = int(start_trace + d['fsd'] * 11 * d['epochlen'])
		if vid_flag:
			vid_start = timestamp_df.index[timestamp_df['Offset Time']>(i*d['epochlen'])][0]
			vid_end = timestamp_df.index[timestamp_df['Offset Time']<((i*d['epochlen'])+(d['epochlen']*3))][-1]
		SWS_utils.update_raw_trace(line1, line2, line4, marker1, marker2, fig, fig2, start_trace, end_trace,
								   this_eeg, DTh, d['emg'], this_emg, realtime, d['fsd'], d['epochlen'])
		color_dict = {'0': 'white',
					  '1': 'green',
					  '2': 'blue',
					  '3': 'red',
					  '4': 'purple'}

		if math.isnan(State[i-1]):
			rect = patch.Rectangle((realtime[start_trace], 0),
								  (epochlen), 1, color=color_dict[str(0)])
		else:
			rect = patch.Rectangle((realtime[start_trace], 0),
							   (epochlen), 1, color=color_dict[str(int(State[i - 1]))])
		ax6.add_patch(rect)
		fig.show()
		fig2.show()
		button = False
		while not button:
			button = fig2.waitforbuttonpress()
			print(f'button: {button}')
			if not button:
				print('you clicked')
				if d['vid']:
					SWS_utils.pull_up_movie(cap, fps, vid_start, vid_end, this_video, d['epochlen'])
				else:
					print('...but you do not have videos available')
		global key_stroke
		State[i] = key_stroke
		fig2.canvas.flush_events()
		fig.canvas.flush_events()
		np.save(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)

	print('DONE SCORING')
	cap.release()
	plt.close('all')
	last_state = int(input('Enter the last state: '))
	State[-2:] = last_state
	np.save(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)
	return State

def update_model(d, this_eeg, FeatureDict, a, EEG_datetime):
	# Feed the data to retrain a model.
	# Using EMG data by default. (No video for now)
	FeatureDict = SWS_utils.adjust_movement(FeatureDict, d['movement'])
	final_features = list(FeatureDict.keys())
	data = list(FeatureDict.values())

	FeatureDict['EMGvar'][np.isnan(FeatureDict['EMGvar'])] = 0
	df_additions = pd.DataFrame(FeatureDict)
	df_additions[pd.isnull(FeatureDict['EMGvar'])] = 0


	Sleep_Model = SWS_utils.update_sleep_df(d['model_dir'], d['mod_name'], df_additions)
	jobname, x_features = SWS_utils.load_joblib(FeatureDict, d['emg'], d['movement'], d['mod_name'])
	Sleep_Model = Sleep_Model.drop(index=np.where(Sleep_Model['EMGvar'].isin(['nan']))[0])
	SWS_utils.retrain_model(Sleep_Model, x_features, d['model_dir'], jobname)


def display_and_fix_scoring(d, this_eeg, a, h, this_emg, State_input, is_predicted, clf, Features, this_video,
	EEG_datetime, v = None, movement_df = None):
	plt.ion()
	i = 0
	this_bin = 1*d['fsd']*d['epochlen']
	realtime = np.arange(np.size(this_eeg))/d['fsd']

	start_trace = int(i * d['fsd'] * d['epochlen'])
	end_trace = int(start_trace + d['fsd'] * 11 * d['epochlen'])
	LFP_ylim = 5

	if d['vid']:
		print('Loading video now, this might take a second....')
		cap, timestamp_df, fps = SWS_utils.load_video(this_video, a, d['Acquisition'], d['savedir'])
		this_timestamp = SWS_utils.pulling_timestamp(timestamp_df, EEG_datetime, this_eeg, d['fsd'])

	print('loading the theta ratio...')
	DTh = SWS_utils.load_bands(this_eeg, d['fsd'])
	fig2, (ax4, ax5, ax7) = plt.subplots(nrows=3, ncols=1, figsize=(11, 6))
	line1, line2, line4 = SWS_utils.pull_up_raw_trace(ax4, ax5, ax7, d['emg'], start_trace, end_trace, realtime,
															 this_eeg, d['fsd'], LFP_ylim, DTh,
															 d['epochlen'], this_emg)

	fig, ax1, ax2, axx = SWS_utils.create_prediction_figure(State_input, is_predicted, clf, 
			Features, d['fsd'], this_eeg, this_emg, realtime, d['epochlen'], start_trace, end_trace, 
			d['Maximum_Frequency'], d['Minimum_Frequency'], movement_flag = d['movement'], v = v)

	marker1, marker2 = SWS_utils.make_marker(ax1, ax2, this_bin, realtime, d['fsd'], d['epochlen'])


	plt.ion()	
	State = copy.deepcopy(State_input)
	#init cursor and it's libraries from SW_Cursor.py
	cursor = Cursor(ax1, ax2, axx)	

	cID = fig.canvas.mpl_connect('button_press_event', cursor.on_click)


	cID4 = fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
	cID4 = fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)

	#Ok so I think that the quotes is the specific event to trigger and the second arg is the function to run when that happens?
	cID2 = fig.canvas.mpl_connect('axes_enter_event', cursor.in_axes)
	cID3 = fig.canvas.mpl_connect('key_press_event', cursor.on_press)



	#This is the loop that manages the interface
	plt.show()
	DONE = False
	while not DONE:
		plt.waitforbuttonpress()

		if cursor.replot:
			print("Replot of fig 1. called!")

			# Call a replot of the graph here

			print('start = '+str(start_trace))
			print('end = '+str(end_trace))
			print('fsd = '+str(d['fsd']))
			print('sindex = '+str((start_trace+(cursor.replotx*d['fsd']))))
			print('eindex = ' + str((end_trace + (cursor.replotx * d['fsd']))))

			#Bumping up by x3 to test if this is all that's needed
			SWS_utils.update_raw_trace(line1, line2, line4, marker1, marker2, fig, fig2, int(start_trace+(cursor.replotx*d['fsd'])), 
							int(end_trace+(cursor.replotx*d['fsd'])), this_eeg, DTh, 
							d['emg'], this_emg, realtime, d['fsd'], d['epochlen'])
			if d['vid']:
				if cursor.replotx-d['epochlen'] < 0:
					print('No video available for this bin')
				else:
					vid_start = int(this_timestamp.index[this_timestamp['Offset_Time']>(cursor.replotx-d['epochlen'])][0])
					vid_end = int(this_timestamp.index[this_timestamp['Offset_Time']<((cursor.replotx)+(d['epochlen']*2))][-1])
					# if (vid_start < this_timestamp.index[0]) or (vid_end< this_timestamp.index[0]):
					# 	print('No video available for this bin')
					# else:
					SWS_utils.pull_up_movie(cap, vid_start, vid_end, 
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
			fig.canvas.draw()
			# new_state = int(input('What state should these be?: '))
			try:
				new_state = int(input('What state should these be?: '))
			except:
				new_state = int(input('What state should these be?: '))
			SWS_utils.correct_bins(start_bin, end_bin, ax2, new_state)
			fig.canvas.draw()
			State[start_bin:end_bin+1] = new_state
			cursor.bins = []
			cursor.change_bins = False
		if cursor.DONE:
			DONE = True

	print('successfully left GUI')
	cv2.destroyAllWindows()
	plt.close('all')
	np.save(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)

	return State


def start_swscoring(filename_sw, d):
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
	if d['emg']:
		downsampEMG = np.load(os.path.join(d['savedir'],'downsampEMG_Acq'+str(a)+'.npy'))
	acq_len = np.size(downsampEEG)/d['fsd'] # fs: sampling rate, fsd: downsampled sampling rate
	hour_segs = math.ceil(acq_len/3600) # acq_len in seconds, convert to hours
	print('This acquisition has ' +str(hour_segs)+ ' segments.')
	AD_file = os.path.join(d['rawdat_dir'], 'AD' + str(d['EEG channel']) + '_'+str(a)+'.mat')
	EEG_datestring = time.ctime(os.path.getmtime(AD_file))
	ts_format = '%a %b %d %H:%M:%S %Y'
	EEG_datetime = datetime.strptime(EEG_datestring, ts_format)
	for h in np.arange(hour_segs):
		# FeatureDict = {}
		this_eeg = np.load(os.path.join(d['savedir'], 'downsampEEG_Acq'+str(a) + '_hr' + str(h)+ '.npy'))
		if d['emg']:
			this_emg = np.load(os.path.join(d['savedir'],'downsampEMG_Acq'+str(a) + '_hr' + str(h)+ '.npy'))
		else:
			this_emg = None
		# chop off the remainder that does not fit into the 4s epoch
		seg_len = np.size(this_eeg)/d['fsd']
		nearest_epoch = math.floor(seg_len/d['epochlen'])
		new_length = int(nearest_epoch*d['epochlen']*d['fsd'])
		this_eeg = this_eeg[0:new_length]

		FeatureDict = SWS_utils.build_feature_dict(this_eeg, d['fsd'], d['epochlen'], this_emg = this_emg)
		this_video, v, this_motion = SWS_utils.initialize_vid_and_move(d, a, EEG_datetime, acq_len, this_eeg)
		FeatureDict['Velocity'] = v[0]
		FeatureDict['animal_name'] = np.full(np.size(FeatureDict['delta_pre']), d['mouse_name'])

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
				color_dict = {'0': 'white',
							  '1': 'green',
							  '2': 'blue',
							  '3': 'red',
							  '4': 'purple'}

				State = display_and_fix_scoring(d, this_eeg, a, h, this_emg, State, False, None,
										None, this_video, EEG_datetime, v = v, movement_df = this_motion)
				if np.any(State == 0):
					print('The following bins are not scored: \n' + str(np.where(State == 0)[0])  )
					zero_check = input('Do you want to go back and fix this right now? (y/n)' ) == 'y'
					if zero_check:
						State = display_and_fix_scoring(d, this_eeg, a, h, this_emg, State, False, None,
										None, this_video, EEG_datetime, v = v, movement_df = this_motion)					
					else:
						print('Ok, but please do not update the model until you fix them')
				FeatureDict['State'] = State
				update = input('Do you want to update the model?: y/n ') == 'y'
				if update:
					update_model(d, this_eeg, FeatureDict, a, EEG_datetime)					
					model_log(d['modellog_dir'], 0, d['species'], d['mouse_name'], d['mod_name'], a)
				logq = input('Do you want to update your personal log?: y/n ') == 'y'
				if logq:
					personal_log(d['personallog_dir'], d['mouse_name'], d['savedir'], a)

			except FileNotFoundError:
				# if the file is a brand new one for scoring
				print("There is no existing scoring.")

		else:
			model = input('Use a random forest? y/n: ') == 'y'

			if model:
				if d['emg']:
					jobname = d['mod_name'] + '_EMG'
					print("EMG flag on")
				else:
					x_features.remove('EMG')
					jobname = d['mod_name'] + '_no_EMG'
					print('Just so you know...this model has no EMG')
				if d['movement']:
					jobname = jobname + '_movement'
				else:
					jobname = jobname + '_no_movement'
				jobname = jobname + '.joblib'
				os.chdir(d['model_dir'])
				try:
					clf = joblib.load(jobname)
				except FileNotFoundError:
					print("You don't have a model to work with.")
					print("Run \"python train_model.py\" before scoring to obtain your very first model.")
					return

				# feature list
				Features = SWS_utils.prepare_feature_data(FeatureDict, d['emg'])

				Predict_y = clf.predict(Features)
				Predict_y = SWS_utils.fix_states(Predict_y)
				np.save(os.path.join(d['savedir'], 'model_prediction_Acq' + str(a) + '_hr' + str(h) + '.npy'), Predict_y)

				Predict_y = display_and_fix_scoring(d, this_eeg, a, h, this_emg, Predict_y, True, clf,
					Features, this_video, EEG_datetime, v = v, movement_df = this_motion)
				FeatureDict['State'] = Predict_y
				FeatureDict['animal_name'] = np.full(np.size(FeatureDict['delta_pre']), d['mouse_name'])

				plt.close('all')
				# Store the result.
				np.save(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), Predict_y)
				update = input('Do you want to update the model?: y/n ') == 'y'
				if update:
					update_model(d, this_eeg, FeatureDict, a, EEG_datetime)
					model_log(d['modellog_dir'], 1, d['species'], d['mouse_name'], d['mod_name'], a)
				logq = input('Do you want to update your personal log?: y/n ') == 'y'
				if logq:
					personal_log(d['personallog_dir'], d['mouse_name'], d['savedir'], a)
			# No model code
			else:
				State = manual_scoring(d, a, this_eeg, this_emg, this_video, h, EEG_datetime, v = v)
				update = input('Do you want to update the model?: y/n ') == 'y'
				if update:
					update_model(d, this_eeg, FeatureDict, a, EEG_datetime)
					model_log(d['modellog_dir'], 2, d['species'], d['mouse_name'], d['mod_name'], a)
				logq = input('Do you want to update your personal log? (y/n) ') == 'y'
				if logq:
					personal_log(d['personallog_dir'], d['mouse_name'], d['savedir'], a)



def load_data_for_sw(filename_sw):
	with open(filename_sw, 'r') as f:
		d = json.load(f)

	start_swscoring(filename_sw, d)

def build_model(filename_sw):
	with open(filename_sw, 'r') as f:
		d = json.load(f)

	print('this code is supressing warnings')
	warnings.filterwarnings("ignore")
	print('These are the available acquisitions: '+ str(d['Acquisition']))
	these_acqs = input('Which acqusitions do you want to use in the model?').split()

	for a in these_acqs:
		print('Loading EEG and EMG....')
		downsampEEG = np.load(os.path.join(d['savedir'],'downsampEEG_Acq'+str(a)+'.npy'))
		if d['emg']:
			downsampEMG = np.load(os.path.join(d['savedir'],'downsampEMG_Acq'+str(a)+'.npy'))
		acq_len = np.size(downsampEEG)/d['fsd'] # fs: sampling rate, fsd: downsampled sampling rate
		AD_file = os.path.join( d['rawdat_dir'], 'AD0_'+str(a)+'.mat')
		EEG_datestring = time.ctime(os.path.getmtime(AD_file))
		ts_format = '%a %b %d %H:%M:%S %Y'
		EEG_datetime = datetime.strptime(EEG_datestring, ts_format)
		this_eeg = np.load(os.path.join(d['savedir'], 'downsampEEG_Acq'+str(a) + '_hr0.npy'))
		if d['emg'] == 1:
			this_emg = np.load(os.path.join(d['savedir'],'downsampEMG_Acq'+str(a) + '_hr0.npy'))
		else:
			this_emg = None
		# chop off the remainder that does not fit into the 4s epoch
		seg_len = np.size(this_eeg)/d['fsd']
		nearest_epoch = math.floor(seg_len/d['epochlen'])
		new_length = int(nearest_epoch*d['epochlen']*d['fsd'])
		this_eeg = this_eeg[0:new_length]

		FeatureDict = SWS_utils.build_feature_dict(this_eeg, d['fsd'], d['epochlen'], this_emg = this_emg)
		this_video, v, this_motion = SWS_utils.initialize_vid_and_move(d, a, EEG_datetime, acq_len, this_eeg)
		FeatureDict['Velocity'] = v[0]
		FeatureDict['animal_name'] = np.full(np.size(FeatureDict['delta_pre']), d['mouse_name'])
		try:
			State = np.load(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr0.npy'))
			wrong, = np.where(np.isnan(State))
			State[wrong] = 0
			State = display_and_fix_scoring(d, this_eeg, a, 0, this_emg, State, False, None,
									None, this_video, EEG_datetime, v = v, movement_df = this_motion)
			FeatureDict['State'] = State
			keep = input('Do you want this to be part of the model? (y/n)') == 'y'
			if keep:
				update_model(d, this_eeg, FeatureDict, a, EEG_datetime)
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
