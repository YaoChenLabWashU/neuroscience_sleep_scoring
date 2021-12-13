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
from joblib import dump, load
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


def manual_scoring(extracted_dir, a, acq, this_eeg, fsd, epochlen, emg_flag, this_emg, 
	vid_flag, this_video, h, movement_flag, bonsai_v, maxfreq, minfreq, EEG_datetime, v = None):
	# Manually score the entire file.
	plt.ion()
	fig, (ax1, ax2, ax4, axx) = plt.subplots(nrows=4, ncols=1, figsize=(11, 6))
	fig2, ax5, ax6 = SWS_utils.create_scoring_figure(extracted_dir, a, this_eeg, fsd, maxfreq, minfreq, movement_flag = movement_flag, v = v)
	
	cID2 = fig.canvas.mpl_connect('key_press_event', on_press)
	cID3 = fig2.canvas.mpl_connect('key_press_event', on_press)
	i = 0
	this_bin = 1*fsd*epochlen
	realtime = np.arange(np.size(this_eeg)) / fsd

	start_trace = int(i * fsd * epochlen)
	end_trace = int(start_trace + fsd * 11 * epochlen)
	LFP_ylim = 5

	DTh = SWS_utils.load_bands(this_eeg, fsd)
	if vid_flag:
		print('Loading video now, this might take a second....')
		cap, timestamp_df, fps = SWS_utils.load_video(this_video, a, acq, extracted_dir)
		timestamp_df = SWS_utils.pulling_timestamp(timestamp_df, EEG_datetime, this_eeg, fsd)


	line1, line2, line4, line5 = SWS_utils.raw_scoring_trace(ax1, ax2, ax4, axx, 
															 emg_flag, start_trace, end_trace, realtime, this_eeg, fsd,
															 LFP_ylim, DTh, epochlen, this_emg)

	marker1, marker2 = SWS_utils.make_marker(ax5, None, this_bin, realtime, fsd, epochlen, num_markers = 1)

	fig.show()
	fig2.show()
	fig.tight_layout()
	fig2.tight_layout()

	plt.show()

	try:
		# if some portion of the file has been previously scored
		State = np.load(os.path.join(extracted_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'))
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
			start = int(count * fsd * epochlen)
			rect = patch.Rectangle((realtime[start + (epochlen * fsd)], 0),
								   (epochlen), 1, color=color_dict[str(int(color))])
			ax6.add_patch(rect)
		fig2.show()

	except FileNotFoundError:
		# if the file is a brand new one for scoring
		State = np.zeros(int(np.size(this_eeg) / fsd / epochlen))
		s = np.arange(1, np.size(State) - 1)
		first_state = int(input('Enter the first state: '))
		State[0] = first_state

	for i in s[:-3]:
		# input('press enter or quit')
		print(f'here. index: {i}')
		start_trace = int(i * fsd * epochlen)
		end_trace = int(start_trace + fsd * 11 * epochlen)
		if vid_flag:
			vid_start = timestamp_df.index[timestamp_df['Offset Time']>(i*epochlen)][0]
			vid_end = timestamp_df.index[timestamp_df['Offset Time']<((i*epochlen)+(epochlen*3))][-1]
		SWS_utils.update_raw_trace(line1, line2, line4, marker1, marker2, fig, fig2, start_trace, end_trace,
								   this_eeg, DTh, emg_flag, this_emg, realtime, fsd, epochlen)
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
				if vid_flag:
					SWS_utils.pull_up_movie(cap, fps, vid_start, vid_end, this_video, epochlen)
				else:
					print('...but you do not have videos available')
		global key_stroke
		State[i] = key_stroke
		fig2.canvas.flush_events()
		fig.canvas.flush_events()
		np.save(os.path.join(extracted_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)

	print('DONE SCORING')
	cap.release()
	plt.close('all')
	last_state = int(input('Enter the last state: '))
	State[-2:] = last_state
	np.save(os.path.join(extracted_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)
	return State

def update_model(this_eeg, fsd, epochlen, animal_name, State, delta_pre, delta_pre2, delta_pre3, delta_post,
		delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
		theta_post3,
		EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, thet_delt, EEGfire, EEGamp, EEGmax,
		EEGmean, EMGamp, model_dir, mod_name, emg_flag, movement_flag, vid_flag, video_dir, acq, a, bonsai_v, 
		EEG_datetime, extracted_dir):
	# Feed the data to retrain a model.
	# Using EMG data by default. (No video for now)
	if movement_flag:
		this_video, v, this_motion = SWS_utils.initialize_vid_and_move(bonsai_v, vid_flag, movement_flag, video_dir, a, 
			acq, this_eeg, fsd, EEG_datetime, extracted_dir)
		if np.shape(v)[1] > 900:
			v_reshape = np.reshape(v[0], (-1,epochlen))
			mean_v = np.mean(v_reshape, axis = 1)
			mean_v[np.isnan(mean_v)] = 0
		elif np.shape(v)[1] < 900:
			diff = 900 - np.shape(v)[1]
			nans = np.empty(diff)
			nans[:] = 0
			mean_v = np.concatenate((v[0], nans))
		else:
			mean_v = v[0]
	else:
		mean_v = np.zeros(900)

	final_features = ['Animal_Name', 'State', 'delta_pre', 'delta_pre2',
					'delta_pre3', 'delta_pot', 'delta_post2', 'delta_post3', 'EEGdelta', 'theta_pre',
					'theta_pre2', 'theta_pre3',
					'theta_post', 'theta_post2', 'theta_post3', 'EEGtheta', 'EEGalpha', 'EEGbeta',
					'EEGgamma', 'EEGnarrow', 'nb_pre', 'theta/delta', 'EEGfire', 'EEGamp', 'EEGmax',
					'EEGmean', 'EMG', 'Movement']
	data = [animal_name, State, delta_pre, delta_pre2, delta_pre3, delta_post,
		delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
		theta_post3,
		EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, thet_delt, EEGfire, EEGamp, EEGmax,
		EEGmean]

	EMGamp[np.isnan(EMGamp)] = 0
	data.append(EMGamp)
	mean_v[np.isnan(mean_v)] = 0
	data.append(mean_v)
	data_dict = dict(zip(final_features, data))
	df_additions = pd.DataFrame(data_dict)
	df_additions[pd.isnull(EMGamp)] = 0


	Sleep_Model = SWS_utils.update_sleep_model(model_dir, mod_name, df_additions)
	jobname, x_features = SWS_utils.load_joblib(final_features, emg_flag, movement_flag, mod_name)
	Sleep_Model = Sleep_Model.drop(index=np.where(Sleep_Model['EMG'].isin(['nan']))[0])
	SWS_utils.retrain_model(Sleep_Model, x_features, model_dir, jobname)


def display_and_fix_scoring(fsd, epochlen, this_eeg, extracted_dir, a, acq, h, emg_flag, 
	movement_flag, this_emg, State_input, is_predicted, clf, Features, vid_flag, this_video, bonsai_v, 
	maxfreq, minfreq, EEG_datetime, v = None, movement_df = None):
	plt.ion()
	i = 0
	this_bin = 1*fsd*epochlen
	realtime = np.arange(np.size(this_eeg)) / fsd
	this_bin = 1*fsd*epochlen

	start_trace = int(i * fsd * epochlen)
	end_trace = int(start_trace + fsd * 11 * epochlen)
	LFP_ylim = 5

	if vid_flag:
		print('Loading video now, this might take a second....')
		cap, timestamp_df, fps = SWS_utils.load_video(this_video, a, acq, extracted_dir)
		this_timestamp = SWS_utils.pulling_timestamp(timestamp_df, EEG_datetime, this_eeg, fsd)

	print('loading the theta ratio...')
	DTh = SWS_utils.load_bands(this_eeg, fsd)
	fig2, (ax4, ax5, ax7) = plt.subplots(nrows=3, ncols=1, figsize=(11, 6))
	line1, line2, line4 = SWS_utils.pull_up_raw_trace(ax4, ax5, ax7, emg_flag, start_trace, end_trace, realtime,
															 this_eeg, fsd, LFP_ylim, DTh,
															 epochlen, this_emg)

	fig, ax1, ax2, axx = SWS_utils.create_prediction_figure(State_input, is_predicted, clf, 
			Features, fsd, this_eeg, this_emg, realtime, epochlen, start_trace, end_trace, maxfreq, minfreq, movement_flag = movement_flag, v = v)

	marker1, marker2 = SWS_utils.make_marker(ax1, ax2, this_bin, realtime, fsd, epochlen)


	plt.ion()	
	State = copy.deepcopy(State_input)
	#init cursor and it's libraries from SW_Cursor.py
	#cursor = Cursor(ax1, ax2, ax3, axx)
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
			print('fsd = '+str(fsd))
			print('sindex = '+str((start_trace+(cursor.replotx*fsd))))
			print('eindex = ' + str((end_trace + (cursor.replotx * fsd))))

			#Bumping up by x3 to test if this is all that's needed
			SWS_utils.update_raw_trace(line1, line2, line4, marker1, marker2, fig, fig2, int(start_trace+(cursor.replotx*fsd)), 
							int(end_trace+(cursor.replotx*fsd)), this_eeg, DTh, 
							emg_flag, this_emg, realtime, fsd, epochlen)
			if vid_flag:
				if cursor.replotx-epochlen < 0:
					print('No video available for this bin')
				else:
					vid_start = int(this_timestamp.index[this_timestamp['Offset_Time']>(cursor.replotx-epochlen)][0])
					vid_end = int(this_timestamp.index[this_timestamp['Offset_Time']<((cursor.replotx)+(epochlen*2))][-1])
					if (vid_start < this_timestamp.index[0]) or (vid_end< this_timestamp.index[0]):
						print('No video available for this bin')
					else:
						SWS_utils.pull_up_movie(cap, fps, vid_start, vid_end, this_video, epochlen)


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
	np.save(os.path.join(extracted_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)

	return State


def start_swscoring(filename_sw, extracted_dir,  rawdat_dir, epochlen, fsd, emg_flag, vid_flag, 
	movement_flag, mouse_name, animal, model_dir, mod_name, modellog_dir, personallog_dir, acq, video_dir, bonsai_v,
	maxfreq, minfreq):
	# mostly for deprecated packages
	print('this code is supressing warnings')
	warnings.filterwarnings("ignore")

	print('These are the available acquisitions: '+ str(acq))

	a = input('Which acqusition do you want to score?')

	print('Loading EEG and EMG....')
	downsampEEG = np.load(os.path.join(extracted_dir,'downsampEEG_Acq'+str(a)+'.npy'))
	if emg_flag:
		downsampEMG = np.load(os.path.join(extracted_dir,'downsampEMG_Acq'+str(a)+'.npy'))
	acq_len = np.size(downsampEEG)/fsd # fs: sampling rate, fsd: downsampled sampling rate
	hour_segs = math.ceil(acq_len/3600) # acq_len in seconds, convert to hours
	print('This acquisition has ' +str(hour_segs)+ ' segments.')
	AD_file = os.path.join(rawdat_dir, 'AD0_'+str(a)+'.mat')
	EEG_datestring = time.ctime(os.path.getmtime(AD_file))
	ts_format = '%a %b %d %H:%M:%S %Y'
	EEG_datetime = datetime.strptime(EEG_datestring, ts_format)
	for h in np.arange(hour_segs):
		this_eeg = np.load(os.path.join(extracted_dir, 'downsampEEG_Acq'+str(a) + '_hr' + str(h)+ '.npy'))
		if emg_flag == 1:
			this_emg = np.load(os.path.join(extracted_dir,'downsampEMG_Acq'+str(a) + '_hr' + str(h)+ '.npy'))
		else:
			this_emg = None
		# chop off the remainder that does not fit into the 4s epoch
		seg_len = np.size(this_eeg)/fsd
		nearest_epoch = math.floor(seg_len/epochlen)
		new_length = int(nearest_epoch*epochlen*fsd)
		this_eeg = this_eeg[0:new_length]

		this_video, v, this_motion = SWS_utils.initialize_vid_and_move(bonsai_v, vid_flag, movement_flag, video_dir, a, 
			acq, this_eeg, fsd, EEG_datetime, extracted_dir)

		os.chdir(extracted_dir)
		print('Generating EMG vectors...')
		if emg_flag:
			EMGamp, EMGmax, EMGmean = SWS_utils.generate_signal(this_emg, epochlen, fsd)
		else:
			EMGamp = False


		print('Generating EEG vectors...')
		EEGamp, EEGmax, EEGmean = SWS_utils.generate_signal(this_eeg, epochlen, fsd)

		print('Extracting delta bandpower...') # non RAM (slow wave) sleep value | per epoch
		EEGdelta, idx_delta = SWS_utils.bandPower(0.5, 4, this_eeg, epochlen, fsd)

		print('Extracting theta bandpower...') # awake / REM sleep
		EEGtheta, idx_theta = SWS_utils.bandPower(5, 8, this_eeg, epochlen, fsd)

		print('Extracting alpha bandpower...') # awake / RAM; not use a lot
		EEGalpha, idx_alpha = SWS_utils.bandPower(8, 12, this_eeg, epochlen, fsd)

		print('Extracting beta bandpower...') # awake; not use a lot
		EEGbeta, idx_beta = SWS_utils.bandPower(20, 30, this_eeg, epochlen, fsd)

		print('Extracting gamma bandpower...') # only awake
		EEGgamma, idx_gamma = SWS_utils.bandPower(30, 80, this_eeg, epochlen, fsd)

		print('Extracting narrow-band theta bandpower...') # broad-band theta
		EEG_broadtheta, idx_broadtheta = SWS_utils.bandPower(2, 16, this_eeg, epochlen, fsd)

		print('Boom. Boom. FIYA POWER...')
		EEGfire, idx_fire = SWS_utils.bandPower(4, 20, this_eeg, epochlen, fsd)

		EEGnb = EEGtheta / EEG_broadtheta # narrow-band theta
		# delt_thet = EEGdelta / EEGtheta # ratio; esp. important
		thet_delt = EEGtheta / EEGdelta

		EEGdelta = SWS_utils.normalize(EEGdelta)
		EEGalpha = SWS_utils.normalize(EEGalpha)
		EEGbeta = SWS_utils.normalize(EEGbeta)
		EEGgamma = SWS_utils.normalize(EEGbeta)
		EEGnb = SWS_utils.normalize(EEGnb)
		EEGtheta = SWS_utils.normalize(EEGtheta)
		EEGfire = SWS_utils.normalize(EEGfire)
		thet_delt = SWS_utils.normalize(thet_delt)

		# frame shifting
		delta_post, delta_pre = SWS_utils.post_pre(EEGdelta, EEGdelta)
		theta_post, theta_pre = SWS_utils.post_pre(EEGtheta, EEGtheta)
		delta_post2, delta_pre2 = SWS_utils.post_pre(delta_post, delta_pre)
		theta_post2, theta_pre2 = SWS_utils.post_pre(theta_post, theta_pre)
		delta_post3, delta_pre3 = SWS_utils.post_pre(delta_post2, delta_pre2)
		theta_post3, theta_pre3 = SWS_utils.post_pre(theta_post2, theta_pre2)
		nb_post, nb_pre = SWS_utils.post_pre(EEGnb, EEGnb)

###--------------------------------This is where the model stuff will go--------------------###

		animal_name = np.full(np.size(delta_pre), mouse_name)


		check = input('Do you want to check and fix existing scoring (c) or score new dataset (s)?: c/s ')
		while check != 'c' and check != 's':
			check = input(
				'Only c/s is accepted. Do you want to check and fix existing scoring (c) or score new dataset (s)?: c/s ')
		if check == 'c':
			try:
				# if some portion of the file has been previously scored
				State = np.load(os.path.join(extracted_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'))
				wrong, = np.where(np.isnan(State))
				State[wrong] = 0
				s, = np.where(State == 0)
				color_dict = {'0': 'white',
							  '1': 'green',
							  '2': 'blue',
							  '3': 'red',
							  '4': 'purple'}

				State = display_and_fix_scoring(fsd, epochlen, this_eeg, extracted_dir, a, acq, h, emg_flag, movement_flag, this_emg, State, False, None,
										None, vid_flag, this_video, bonsai_v, maxfreq, minfreq, EEG_datetime, v = v, movement_df = this_motion)
				if np.any(State == 0):
					print('The following bins are not scored: \n' + str(np.where(State == 0)[0])  )
					zero_check = input('Do you want to go back and fix this right now? (y/n)' ) == 'y'
					if zero_check:
						State = display_and_fix_scoring(fsd, epochlen, this_eeg, extracted_dir, a, acq, h, emg_flag, movement_flag, this_emg, State, False, None,
										None, vid_flag, this_video, bonsai_v, maxfreq, minfreq, EEG_datetime, v = v, movement_df = this_motion)					
					else:
						print('Ok, but please do not update the model until you fix them')
				update = input('Do you want to update the model?: y/n ') == 'y'
				if update:
					update_model(this_eeg, fsd, epochlen, animal_name, State, delta_pre, delta_pre2, delta_pre3, delta_post,
							 delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
							 theta_post3, EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, thet_delt, EEGfire, EEGamp, EEGmax,
							 EEGmean, EMGamp, model_dir, mod_name, emg_flag, movement_flag, vid_flag, video_dir, acq, a, bonsai_v, 
							 EEG_datetime, extracted_dir)
					model_log(modellog_dir, 0, animal, mouse_name, mod_name, a)
				logq = input('Do you want to update your personal log?: y/n ') == 'y'
				if logq:
					personal_log(personallog_dir, mouse_name, extracted_dir, a)

			except FileNotFoundError:
				# if the file is a brand new one for scoring
				print("There is no existing scoring.")

		else:
			model = input('Use a random forest? y/n: ') == 'y'

			if model:
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
				os.chdir(model_dir)
				try:
					clf = load(jobname)
				except FileNotFoundError:
					print("You don't have a model to work with.")
					print("Run \"python train_model.py\" before scoring to obtain your very first model.")
					return

				# feature list
				FeatureList = []
				nans = np.full(np.shape(animal_name), np.nan)
				if emg_flag:
					FeatureList = [delta_pre, delta_pre2, delta_pre3, delta_post,
						delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
						theta_post3,EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, thet_delt, EEGfire, EEGamp, EEGmax,
						EEGmean, EMGamp]

				else:
					FeatureList = [delta_pre, delta_pre2, delta_pre3, delta_post, delta_post2, delta_post3,
								   EEGdelta,
								   theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2, theta_post3,
								   EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, thet_delt, EEGfire, EEGamp, EEGmax,
								   EEGmean, nans]
				if movement_flag:
					this_video, v, this_motion = SWS_utils.initialize_vid_and_move(bonsai_v, vid_flag, movement_flag, video_dir, a, 
						acq, this_eeg, fsd, EEG_datetime, extracted_dir)
					if np.shape(v)[1] > 900:
						v_reshape = np.reshape(v[0], (-1,epochlen))
						mean_v = np.mean(v_reshape, axis = 1)
						mean_v[np.isnan(mean_v)] = 0
					elif np.shape(v)[1] < 900:
						diff = 900 - np.shape(v)[1]
						nans = np.empty(diff)
						nans[:] = np.NaN
						mean_v = np.concatenate((v[0], nans))
					else:
						mean_v = v[0]
				else:
					FeatureList.append(nans)
					v = None
				FeatureList.append(mean_v)

				FeatureList_smoothed = []
				for f in FeatureList:
					FeatureList_smoothed.append(signal.medfilt(f, 5))
				Features = np.column_stack((FeatureList_smoothed))
				Features = np.nan_to_num(Features)

				Predict_y = clf.predict(Features)
				Predict_y = SWS_utils.fix_states(Predict_y)
				np.save(os.path.join(extracted_dir, 'model_prediction_Acq' + str(a) + '_hr' + str(h) + '.npy'), Predict_y)

				Predict_y = display_and_fix_scoring(fsd, epochlen, this_eeg, extracted_dir, a, acq, h, emg_flag, 
					movement_flag, this_emg, Predict_y, True, clf, Features, vid_flag, this_video, bonsai_v, maxfreq, minfreq, 
					EEG_datetime, v = v, movement_df = this_motion)


				satisfaction = input('Satisfied?: y/n ') == 'y'
				plt.close('all')

				if satisfaction:
					# Store the result.
					np.save(os.path.join(extracted_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), Predict_y)
				update = input('Do you want to update the model?: y/n ') == 'y'
				if update:
					update_model(this_eeg, fsd, epochlen, animal_name, Predict_y, delta_pre, delta_pre2, delta_pre3, delta_post,
							 delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
							 theta_post3, EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, thet_delt, EEGfire, EEGamp, EEGmax,
							 EEGmean, EMGamp, model_dir, mod_name, emg_flag, movement_flag, vid_flag, video_dir, acq, a, bonsai_v, 
							 EEG_datetime, extracted_dir)
					model_log(modellog_dir, 1, animal, mouse_name, mod_name, a)
				logq = input('Do you want to update your personal log?: y/n ') == 'y'
				if logq:
					personal_log(personallog_dir, mouse_name, extracted_dir, a)
			# No model code
			else:
				State = manual_scoring(extracted_dir, a, acq, this_eeg, fsd, epochlen, emg_flag, 
					this_emg, vid_flag, this_video, h, movement_flag, bonsai_v, maxfreq, minfreq, EEG_datetime, v = v)
				update = input('Do you want to update the model?: y/n ') == 'y'
				if update:
					update_model(this_eeg, fsd, epochlen, animal_name, State, delta_pre, delta_pre2, delta_pre3, delta_post,
								 delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post,
								 theta_post2,
								 theta_post3, EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, thet_delt, EEGfire,
								 EEGamp, EEGmax,
								 EEGmean, EMGamp, model_dir, mod_name, emg_flag, movement_flag, vid_flag, video_dir, acq, a, 
								 bonsai_v, EEG_datetime, extracted_dir)
					model_log(modellog_dir, 2, animal, mouse_name, mod_name, a)
				logq = input('Do you want to update your personal log?: y/n ') == 'y'
				if logq:
					personal_log(personallog_dir, mouse_name, extracted_dir, a)



def load_data_for_sw(filename_sw):
	with open(filename_sw, 'r') as f:
		d = json.load(f)

	extracted_dir = str(d['savedir'])
	epochlen = int(d['epochlen'])
	fsd = int(d['fsd'])
	emg_flag = int(d['emg'])
	vid_flag = int(d['vid'])
	movement_flag = int(d['movement'])
	model_dir = str(d['model_dir'])
	animal = str(d['species'])
	mod_name = str(d['mod_name'])
	modellog_dir = str(d['modellog_dir'])
	personallog_dir = str(d['personallog_dir'])
	mouse_name = str(d['mouse_name'])
	acq = d['Acquisition']
	video_dir = d['video_dir']
	bonsai_v = d['Bonsai Version']
	maxfreq = d['Maximum_Frequency']
	minfreq = d['Minimum_Frequency']
	rawdat_dir = d['rawdat_dir']

	start_swscoring(filename_sw, extracted_dir, rawdat_dir, epochlen, fsd, emg_flag, vid_flag, 
		movement_flag, mouse_name, animal, model_dir, mod_name, modellog_dir, personallog_dir, acq, video_dir, bonsai_v,
		maxfreq, minfreq)


# Arg 1 is the path of the sleep scoring setting json, taken from argv[1]
# Will look something like "Users/evinjaff/Desktop/sleepscoring/Score_Setting.json"
# Type argument is for the type of scoring as an unsinged int
# 0 = corrected, 1 = scored w/ ML model, 2 = scored in legacy mode

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
