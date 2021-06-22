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
import pandas as pd
import warnings
import SWS_utils
import train_model
from SW_Cursor import Cursor


key_stroke = 0

def on_press(event):
	global key_stroke
	if event.key in ['1','2','3']:
		key_stroke = int(event.key)
		print(f'scored: {event.key}')
	elif event.key == 'q':
		print('QUIT')
		plt.close('all')
		sys.exit()
	else:
		key_stroke = np.float('nan')
		print('I did not understand that keystroke; I will mark it white and please come back to fix it.')


def manual_scoring(extracted_dir, a, this_eeg, fsd, epochlen, emg_flag, this_emg, vid_flag, this_video, h):
	# Manually score the entire file.

	# Raw Signal figure
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(11, 6))


	# Spectrogram w/ EEG here
	fig2, ax5, ax6 = SWS_utils.create_scoring_figure(extracted_dir, a, eeg=this_eeg, fsd=fsd)


	# cursor = Cursor(ax5, ax6, ax7)
	cID2 = fig.canvas.mpl_connect('key_press_event', on_press)
	cID3 = fig2.canvas.mpl_connect('key_press_event', on_press)
	i = 0
	start = int(i * fsd * epochlen)
	end = int(start + fsd * 3 * epochlen)
	realtime = np.arange(np.size(this_eeg)) / fsd
	LFP_ylim = 5
	delt = np.load(os.path.join(extracted_dir, 'delt' + str(a) + '_hr' + str(h) + '.npy'))
	thet = np.load(os.path.join(extracted_dir, 'thet' + str(a) + '_hr' + str(h) + '.npy'))

	no_delt_start, = np.where(realtime < delt[1][0])
	no_delt_end, = np.where(realtime > delt[1][-1])
	delt_pad = np.pad(delt[0], (np.size(no_delt_start), np.size(no_delt_end)), 'constant',
					  constant_values=(0, 0))

	no_thet_start, = np.where(realtime < thet[1][0])
	no_thet_end, = np.where(realtime > thet[1][-1])
	thet_pad = np.pad(thet[0], (np.size(no_thet_start), np.size(no_thet_end)), 'constant',
					  constant_values=(0, 0))

	assert np.size(delt_pad) == np.size(this_eeg) == np.size(thet_pad)

	#Pulls raw data into the lines?
	line1, line2, line3, line4 = SWS_utils.pull_up_raw_trace(ax1, ax2, ax3, ax4,
															 emg_flag, start, end, realtime, this_eeg, fsd,
															 LFP_ylim, delt_pad,
															 thet_pad, epochlen, this_emg)
	marker = SWS_utils.make_marker(ax5, end, realtime, fsd, epochlen)

	fig.autoscale()
	fig2.autoscale()

	fig.show()
	fig2.show()
	print("Showing figs")
	fig.tight_layout()
	fig2.tight_layout()
	try:
		# if some portion of the file has been previously scored
		State = np.load(os.path.join(extracted_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'))
		wrong, = np.where(np.isnan(State))
		State[wrong] = 0
		s, = np.where(State == 0)
		color_dict = {'0': 'white',
					  '1': 'green',
					  '2': 'blue',
					  '3': 'red'}
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

	if vid_flag:
		cap = cv2.VideoCapture(this_video)
		fps = cap.get(cv2.CAP_PROP_FPS)
	for i in s[:-3]:
		# input('press enter or quit')
		print(f'here. index: {i}')
		start = int(i * fsd * epochlen)
		end = int(start + fsd * 3 * epochlen)
		if vid_flag:
			vid_start = int(i * fps * epochlen)
			vid_end = int(vid_start + fps * 3 * epochlen)
		SWS_utils.update_raw_trace(line1, line2, line3, line4, marker, fig, fig2, start, end,
								   this_eeg, delt_pad, thet_pad, emg_flag, this_emg, realtime)
		color_dict = {'0': 'white',
					  '1': 'green',
					  '2': 'blue',
					  '3': 'red'}

		if math.isnan(State[i-1]):
			rect = patch.Rectangle((realtime[start], 0),
								  (epochlen), 1, color=color_dict[str(0)])
		else:
			rect = patch.Rectangle((realtime[start], 0),
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
					SWS_utils.pull_up_movie(vid_start, vid_end, this_video, epochlen)
				else:
					print('...but you do not have videos available')
		global key_stroke
		State[i] = key_stroke
		fig2.canvas.flush_events()
		fig.canvas.flush_events()
		np.save(os.path.join(extracted_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)

	print('DONE SCORING')
	plt.close('all')
	last_state = int(input('Enter the last state: '))
	State[-2:] = last_state
	np.save(os.path.join(extracted_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)
	return State


def update_model(animal_name, animal_num, State, delta_pre, delta_pre2, delta_pre3, delta_post,
		 delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
		 theta_post3,
		 EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
		 EEGmean, EMGamp, model_dir, mod_name, emg_flag):
	# Feed the data to retrain a model.
	# Using EMG data by default. (No video for now)
	final_features = ['Animal_Name', 'animal_num', 'State', 'delta_pre', 'delta_pre2',
					  'delta_pre3', 'delta_post', 'delta_post2', 'delta_post3', 'EEGdelta', 'theta_pre',
					  'theta_pre2', 'theta_pre3',
					  'theta_post', 'theta_post2', 'theta_post3', 'EEGtheta', 'EEGalpha', 'EEGbeta',
					  'EEGgamma', 'EEGnarrow', 'nb_pre', 'delta/theta', 'EEGfire', 'EEGamp', 'EEGmax',
					  'EEGmean', 'EMG']
	data = np.vstack(
		[animal_name, animal_num, State, delta_pre, delta_pre2, delta_pre3, delta_post,
		 delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
		 theta_post3,
		 EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
		 EEGmean])

	if np.size(np.where(pd.isnull(EMGamp))[0]) > 0:
		EMGamp[np.isnan(EMGamp)] = 0
	data = np.vstack([data, EMGamp])

	df_additions = pd.DataFrame(columns=final_features, data=data.T)
	Sleep_Model = SWS_utils.update_sleep_model(model_dir, mod_name, df_additions)
	jobname, x_features = SWS_utils.load_joblib(final_features, emg_flag, mod_name)
	Sleep_Model = Sleep_Model.drop(index=np.where(Sleep_Model['EMG'].isin(['nan']))[0])
	SWS_utils.retrain_model(Sleep_Model, x_features, model_dir, jobname)


def display_and_fix_scoring(fsd, epochlen, this_eeg, extracted_dir, a, h, emg_flag, this_emg, State_input, is_predicted, clf, Features, vid_flag, this_video):
	start = 0
	end = int(fsd * 3 * epochlen * 300) # Original formula was: 200 * 3 * 4 = 2400 and that only plotted 12 seconds. So I need to multiply this by 300? There is no data point from the JSON that matches this multiplier so I don't know how to do this
	realtime = np.arange(np.size(this_eeg)) / fsd
	LFP_ylim = 5

	print("realtime: ")
	print(realtime)
	print("\n")


	print('loading delta and theta...')
	delt = np.load(os.path.join(extracted_dir, 'delt' + str(a) + '_hr' + str(h) + '.npy'))
	thet = np.load(os.path.join(extracted_dir, 'thet' + str(a) + '_hr' + str(h) + '.npy'))

	###

	# Plot Theta Wave

	# Data for plotting
	# t = np.arange(0.0, 2.0, 0.01)
	# s = 1 + np.sin(2 * np.pi * t)
	#
	# fig, ax = plt.subplots()
	# ax.plot(t, s)
	#
	# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
	# 	   title='About as simple as it gets, folks')
	# ax.grid()
	#
	# fig.savefig("test.png")
	# plt.show()




	###

	#Is there something wrong here?
	no_delt_start, = np.where(realtime < delt[1][0])
	no_delt_end, = np.where(realtime > delt[1][-1])

	print("no_delt_start: ")
	print(no_delt_start)
	print("\n")

	print("no_delt_end: ")
	print(no_delt_end)
	print("\n")

	delt_pad = np.pad(delt[0], (np.size(no_delt_start), np.size(no_delt_end)), 'constant',
					  constant_values=(0, 0))

	print("delt_pad: ")
	print(delt_pad)
	print("\n")


	no_thet_start, = np.where(realtime < thet[1][0])
	no_thet_end, = np.where(realtime > thet[1][-1])
	thet_pad = np.pad(thet[0], (np.size(no_thet_start), np.size(no_thet_end)), 'constant',
					  constant_values=(0, 0))

	#Figure subplots here, look for variable origin of these
	fig2, (ax4, ax5, ax6, ax7) = plt.subplots(nrows=4, ncols=1, figsize=(11, 6))






	#Look at theese vars
	line1, line2, line3, line4 = SWS_utils.pull_up_raw_trace(ax4, ax5, ax6, ax7, emg_flag, start, end, realtime,
															 this_eeg, fsd, LFP_ylim, delt_pad, thet_pad,
															 epochlen, this_emg)
	fig, ax1, ax2, ax3 = SWS_utils.create_prediction_figure(State_input, is_predicted, clf, Features, fsd, this_eeg)

	plt.ion()
	State = copy.deepcopy(State_input)
	# State[State == 0] = 1
	# State[State == 2] = 2
	# State[State == 5] = 3
	cursor = Cursor(ax1, ax2, ax3)

	cID = fig.canvas.mpl_connect('button_press_event', cursor.on_click)


	cID4 = fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)

	#Ok so I think that the quotes is the specific event to trigger and the second arg is the function to run when that happens?
	cID2 = fig.canvas.mpl_connect('axes_enter_event', cursor.in_axes)
	cID3 = fig.canvas.mpl_connect('key_press_event', cursor.on_press)

	# Try line scaling here:

	plt.autoscale()

	plt.show()
	DONE = False
	while not DONE:
		plt.waitforbuttonpress()
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
			State[start_bin:end_bin] = new_state
			cursor.bins = []
			cursor.change_bins = False
		if cursor.movie_mode and cursor.movie_bin > 0:
			if vid_flag:
				print("Sorry, this function has not been developed yet.")
				# start = int(cursor.movie_bin * 60 * fsd)
				# end = int(((cursor.movie_bin * 60) + 12) * fsd)
				# # end = int(((cursor.movie_bin * 60) + 2) * fsd)
				# marker = SWS_utils.make_marker(ax5, end, realtime, fsd, epochlen)
				# SWS_utils.update_raw_trace(line1, line2, line3, line4, marker, fig, fig2, start, end, this_eeg,
				# 						   delt_pad, thet_pad, emg_flag, this_emg, realtime)
				# fig2.canvas.draw()
				# fig2.tight_layout()
				# # fig2.show()
				# SWS_utils.pull_up_movie(start, end, this_video, epochlen)
				# cursor.movie_bin = 0
			else:
				print("you don't have video, sorry")
		if cursor.DONE:
			DONE = True

	print('successfully left GUI')
	cv2.destroyAllWindows()
	plt.close('all')
	np.save(os.path.join(extracted_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)

	return State


def start_swscoring(filename_sw, extracted_dir,  epochlen, fsd, emg_flag, vid_flag, animal, model_dir, mod_name):
	# mostly for deprecated packages
	print('this code is supressing warnings')
	warnings.filterwarnings("ignore")

	with open(filename_sw, 'r') as f:
			d = json.load(f)

	acq = d['Acquisition']
	video_dir = d['video_dir']

	print('These are the available acquisitions: '+ str(acq))

	a = input('Which acqusition do you want to score?')


	print('Loading EEG and EMG....')
	downsampEEG = np.load(os.path.join(extracted_dir,'downsampEEG_Acq'+str(a)+'.npy'))
	if emg_flag:
		downsampEMG = np.load(os.path.join(extracted_dir,'downsampEMG_Acq'+str(a)+'.npy'))
	acq_len = np.size(downsampEEG)/fsd # fs: sampling rate, fsd: downsampled sampling rate
	hour_segs = math.ceil(acq_len/3600) # acq_len in seconds, convert to hours
	print('This acquisition has ' +str(hour_segs)+ ' segments.')

	for h in np.arange(hour_segs):
		this_eeg = np.load(os.path.join(extracted_dir, 'downsampEEG_Acq'+str(a) + '_hr' + str(h)+ '.npy'))
		if int(d['emg']) == 1:
			this_emg = np.load(os.path.join(extracted_dir,'downsampEMG_Acq'+str(a) + '_hr' + str(h)+ '.npy'))
		else:
			this_emg = None
		# chop off the remainder that does not fit into the 4s epoch
		seg_len = np.size(this_eeg)/fsd
		nearest_epoch = math.floor(seg_len/epochlen)
		new_length = int(nearest_epoch*epochlen*fsd)
		this_eeg = this_eeg[0:new_length]
		if vid_flag:
			this_video = glob.glob(os.path.join(video_dir, '*'+'Video'+str(int(a)-1)+'_filled.mp4'))[0]
			print('using ' + this_video + ' for the video')
		else:
			this_video = None
			print('no video available')

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

		print('Extracting theta bandpower...') # awake / RAM sleep
		EEGtheta, idx_theta = SWS_utils.bandPower(4, 8, this_eeg, epochlen, fsd)

		print('Extracting alpha bandpower...') # awake / RAM; not use a lot
		EEGalpha, idx_alpha = SWS_utils.bandPower(8, 12, this_eeg, epochlen, fsd)

		print('Extracting beta bandpower...') # awake; not use a lot
		EEGbeta, idx_beta = SWS_utils.bandPower(12, 30, this_eeg, epochlen, fsd)

		print('Extracting gamma bandpower...') # only awake
		EEGgamma, idx_gamma = SWS_utils.bandPower(30, 80, this_eeg, epochlen, fsd)

		print('Extracting narrow-band theta bandpower...') # broad-band theta
		EEG_broadtheta, idx_broadtheta = SWS_utils.bandPower(2, 16, this_eeg, epochlen, fsd)

		print('Boom. Boom. FIYA POWER...')
		EEGfire, idx_fire = SWS_utils.bandPower(4, 20, this_eeg, epochlen, fsd)

		EEGnb = EEGtheta / EEG_broadtheta # narrow-band theta
		delt_thet = EEGdelta / EEGtheta # ratio; esp. important

		EEGdelta = SWS_utils.normalize(EEGdelta)
		EEGalpha = SWS_utils.normalize(EEGalpha)
		EEGbeta = SWS_utils.normalize(EEGbeta)
		EEGgamma = SWS_utils.normalize(EEGbeta)
		EEGnb = SWS_utils.normalize(EEGnb)
		EEGtheta = SWS_utils.normalize(EEGtheta)
		EEGfire = SWS_utils.normalize(EEGfire)
		delt_thet = SWS_utils.normalize(delt_thet)

		# frame shifting
		delta_post, delta_pre = SWS_utils.post_pre(EEGdelta, EEGdelta)
		theta_post, theta_pre = SWS_utils.post_pre(EEGtheta, EEGtheta)
		delta_post2, delta_pre2 = SWS_utils.post_pre(delta_post, delta_pre)
		theta_post2, theta_pre2 = SWS_utils.post_pre(theta_post, theta_pre)
		delta_post3, delta_pre3 = SWS_utils.post_pre(delta_post2, delta_pre2)
		theta_post3, theta_pre3 = SWS_utils.post_pre(theta_post2, theta_pre2)
		nb_post, nb_pre = SWS_utils.post_pre(EEGnb, EEGnb)

###--------------------------------This is where the model stuff will go--------------------###

		animal_name = np.full(np.size(delta_pre), animal)
		# Note: The second parameter depends on the actual animal name. For example, if the animal is "KNR00004", we
		# should use "animal[3:]" for "00004"; if the animal is "jaLC_FLiPAKAREEGEMG004", we should use "animal[19:]"
		# for "004".
		nums = [int(i) for i in animal if i.isdigit()]
		animal_num = [str(i) for i in nums]
		animal_num = np.full(np.shape(animal_name), int("".join(animal_num)))


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

				#Should be pretty easy to add a 4th state here TODO

				color_dict = {'0': 'white',
							  '1': 'green',
							  '2': 'blue',
							  '3': 'red'}
				# rendering what has been previously scored
				# for count, color in enumerate(State[:-1]):
				# 	start = int(count * fsd * epochlen)
				# 	rect = patch.Rectangle((realtime[start + (epochlen * fsd)], 0),
				# 						   (epochlen), 1, color=color_dict[str(int(color))])
				# 	ax6.add_patch(rect)
				# fig2.show()

				display_and_fix_scoring(fsd, epochlen, this_eeg, extracted_dir, a, h, emg_flag, this_emg, State, False, None,
										None, vid_flag, this_video)
				update = input('Do you want to update the model?: y/n ') == 'y'
				if update:
					update_model(animal_name, animal_num, State, delta_pre, delta_pre2, delta_pre3, delta_post,
							 delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
							 theta_post3, EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
							 EEGmean, EMGamp, model_dir, mod_name, emg_flag)


			except FileNotFoundError:
				# if the file is a brand new one for scoring
				print("There is no existing scoring.")

		else:
			model = input('Use a random forest? y/n: ') == 'y'

			if model:
				# loading different models
				os.chdir(model_dir)
				try:
					if emg_flag:
						clf = load(mod_name + '_EMG.joblib')
					else:
						clf = load(mod_name + '_no_EMG.joblib')
				except FileNotFoundError:
					print("You don't have a model to work with.")
					print("Run \"python train_model.py\" before scoring to obtain your very first model.")
					return

				# feature list
				FeatureList = []
				nans = np.full(np.shape(animal_name), np.nan)
				if emg_flag:
					FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3, delta_post, delta_post2, delta_post3,
								   EEGdelta,
								   theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2, theta_post3,
								   EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
								   EEGmean, EMGamp]
				else:
					FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3, delta_post, delta_post2, delta_post3,
								   EEGdelta,
								   theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2, theta_post3,
								   EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
								   EEGmean, nans]

				FeatureList_smoothed = []
				for f in FeatureList:
					FeatureList_smoothed.append(signal.medfilt(f, 5))
				Features = np.column_stack((FeatureList_smoothed))
				Features = np.nan_to_num(Features)

				Predict_y = clf.predict(Features)
				Predict_y = SWS_utils.fix_states(Predict_y)
				SWS_utils.create_prediction_figure(Predict_y, True, clf, Features, fsd, this_eeg)

				satisfaction = input('Satisfied?: y/n ') == 'y'
				plt.close('all')

				if satisfaction:
					# Store the result.
					np.save(os.path.join(extracted_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), Predict_y)
				else:
					fix = input('Do you want to fix a few states (f) or manually score the whole thing (m)?: f/m ')
					while fix != 'f' and fix != 'm':
						fix = input('Only f/m is accepted. Do you want to fix a few states (f) or manually score the whole thing (m)?: f/m ')
					if fix == 'f':
						State = display_and_fix_scoring(fsd, epochlen, this_eeg, extracted_dir, a, h, emg_flag, this_emg, Predict_y, True, clf, Features, vid_flag, this_video)
					else:
						print("Manually score the whole thing and update model.")
						State = manual_scoring(extracted_dir, a, this_eeg, fsd, epochlen, emg_flag, this_emg, vid_flag, this_video, h)

					update = input('Do you want to update the model?: y/n ') == 'y'
					if update:
						update_model(animal_name, animal_num, State, delta_pre, delta_pre2, delta_pre3, delta_post,
								 delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
								 theta_post3, EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
								 EEGmean, EMGamp, model_dir, mod_name, emg_flag)

			else:
				State = manual_scoring(extracted_dir, a, this_eeg, fsd, epochlen, emg_flag, this_emg, vid_flag, this_video, h)
				update = input('Do you want to update the model?: y/n ') == 'y'
				if update:
					update_model(animal_name, animal_num, State, delta_pre, delta_pre2, delta_pre3, delta_post,
								 delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post,
								 theta_post2,
								 theta_post3, EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire,
								 EEGamp, EEGmax,
								 EEGmean, EMGamp, model_dir, mod_name, emg_flag)


#Dammit how the hell did I miss this - Evin
# This is where the GD data loads in
def load_data_for_sw(filename_sw):
	with open(filename_sw, 'r') as f:
		d = json.load(f)

	extracted_dir = str(d['savedir'])
	epochlen = int(d['epochlen'])
	fsd = int(d['fsd'])
	emg_flag = int(d['emg'])
	vid_flag = int(d['vid'])
	model_dir = str(d['model_dir'])
	animal = str(d['animal'])
	mod_name = str(d['mod_name'])

	#Debug files
	print("fsd:")
	print(fsd)


	start_swscoring(filename_sw, extracted_dir, epochlen, fsd, emg_flag, vid_flag, animal, model_dir, mod_name)


if __name__ == "__main__":
	args = sys.argv
	assert args[0] == 'New_SWS.py'
	if len(args) < 2:
		print("You need to specify the path of your Score_Settings.json. For instance, run `python New_SWS.py /home/ChenLab_Sleep_Scoring/Score_Settings.json`.")
	elif len(args) > 2:
		print("You only need to specify the path of your Score_Settings.json. For instance, run `python New_SWS.py /home/ChenLab_Sleep_Scoring/Score_Settings.json`.")
	else:
		load_data_for_sw(args[1])
