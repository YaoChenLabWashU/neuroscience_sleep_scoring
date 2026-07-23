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
import datetime as dt
from datetime import datetime
from neuroscience_sleep_scoring.SW_Cursor import Cursor
from neuroscience_sleep_scoring.SW_Cursor import ScoringCursor
import pathlib
import time
from datetime import datetime, timedelta
from scipy import io

key_stroke = 0

# A single hidden Tk root is reused for the state-selection popup so we don't
# construct/tear down a whole Tk() (and its event loop) on every bin correction.
_state_popup_root = None

def _mpl_root():
	"""Return matplotlib's existing Tk root (TkAgg) so every dialog shares ONE Tk
	interpreter. Creating a second tk.Tk() alongside matplotlib's root is the
	classic multi-root deadlock that froze the GUI."""
	try:
		import tkinter as tk
		r = getattr(tk, '_default_root', None)
		if r is not None:
			return r
	except Exception:
		pass
	try:
		import matplotlib.pyplot as plt
		mgr = plt._pylab_helpers.Gcf.get_active()
		if mgr is not None:
			return mgr.canvas.manager.window
	except Exception:
		pass
	return None

def _ask_yes_no(title, message, default_yes=True):
	"""Show a yes/no dialog and return True/False. Uses matplotlib's existing Tk
	root (no competing tk.Tk()); falls back to a terminal prompt if unavailable."""
	try:
		import tkinter as tk
		from tkinter import messagebox
		root = _mpl_root()
		if root is not None:
			return bool(messagebox.askyesno(title, message, parent=root))
		tmp = tk.Tk()
		tmp.withdraw()
		try:
			return bool(messagebox.askyesno(title, message, parent=tmp))
		finally:
			tmp.destroy()
	except Exception:
		suffix = ' (Y/n): ' if default_yes else ' (y/N): '
		resp = input(message + suffix).strip().lower()
		if resp == '':
			return default_yes
		return resp == 'y'

def _state_strip_color(state_value):
	"""Plot color for one epoch's state in the detailed scoring strip. Matches the
	main hypnogram colors (SWS_utils.STATE_COLORS); unknown/NaN -> white."""
	try:
		if np.isnan(state_value):
			return 'white'
	except (TypeError, ValueError):
		pass
	return SWS_utils.STATE_COLORS.get(int(state_value), 'white')

def draw_state_strip(ax_state, State, this_epoch_t, start_trace, end_trace, epochlen):
	"""Draw the sleep state of every epoch visible in the detailed (fig2) window.

	Ported/adapted from Kane's New_SWS_kg.py: the zoomed panels use coordinates
	relative to the current epoch (current epoch at x=0, each epoch epochlen wide).
	One colored rectangle per visible epoch (color from State), with the current
	epoch outlined in yellow. Clicking a rectangle relabels that epoch (see the
	on_state_strip_click handler wired in display_and_fix_scoring)."""
	ax_state.clear()
	cur_idx = int(round(this_epoch_t / epochlen))
	first_rel = int(math.floor(start_trace / epochlen))
	last_rel = int(math.ceil(end_trace / epochlen))
	for rel in range(first_rel, last_rel):
		abs_idx = cur_idx + rel
		x = rel * epochlen
		color = _state_strip_color(State[abs_idx]) if 0 <= abs_idx < len(State) else 'white'
		ax_state.add_patch(patch.Rectangle((x, 0), epochlen, 1, color=color, ec='k', lw=0.5))
	ax_state.add_patch(patch.Rectangle((0, 0), epochlen, 1, fill=False, ec='#fac205', lw=2.5))
	ax_state.set_xlim([start_trace, end_trace])
	ax_state.set_ylim([0, 1])
	ax_state.set_yticks([])
	ax_state.set_ylabel('Sleep\nState')
	ax_state.set_xlabel('Time (s) relative to current epoch (click an epoch to relabel)')

def _recovery_path(d, a, h):
	"""Path of the autosave/recovery file for one acquisition-hour. Kept in a
	'recovery/' subdir (not the top-level savedir) so it is NOT picked up by the
	StatesAcq*.npy scan that marks acquisitions as scored, and so the canonical
	scoring isn't overwritten until the user confirms saving on close. The name
	ends in .npy so np.save doesn't append a second extension."""
	return os.path.join(d['savedir'], 'recovery', 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy')

def choose_state_popup(popup_xy=None):
	"""Ask which state to assign to the selected bins. Returns 1/2/3 or None.

	Reuses one persistent hidden root; each call spawns a lightweight Toplevel
	and blocks on wait_window (same blocking UX as before, far less overhead).
	"""
	try:
		import tkinter as tk
		from tkinter import font as tkfont
	except Exception:
		return None
	# Share matplotlib's Tk root instead of a separate one (avoids the deadlock).
	root = _mpl_root()
	_temp_root = None
	if root is None:
		try:
			root = tk.Tk()
			root.withdraw()
			_temp_root = root
		except Exception:
			return None

	result = {'val': None}
	win = tk.Toplevel(root)
	win.title('Select State')
	win.resizable(False, False)
	win.attributes('-topmost', True)
	bold_font = tkfont.Font(weight='bold')
	tk.Label(win, text='Choose state', font=bold_font).pack(padx=8, pady=6)

	def set_val(v):
		result['val'] = v
		win.destroy()

	tk.Button(win, text='1: Wake', width=16, bg='green', fg='white', font=bold_font,
		command=lambda: set_val(1)).pack(padx=8, pady=4)
	tk.Button(win, text='2: NREM', width=16, bg='blue', fg='white', font=bold_font,
		command=lambda: set_val(2)).pack(padx=8, pady=4)
	tk.Button(win, text='3: REM', width=16, bg='red', fg='white', font=bold_font,
		command=lambda: set_val(3)).pack(padx=8, pady=6)
	# Keyboard shortcuts so the popup can be answered without the mouse.
	win.bind('1', lambda e: set_val(1))
	win.bind('2', lambda e: set_val(2))
	win.bind('3', lambda e: set_val(3))
	if popup_xy is not None:
		x, y = popup_xy
		win.geometry(f'+{int(x)+10}+{int(y)+10}')
	win.grab_set()
	win.focus_force()
	root.wait_window(win)
	if _temp_root is not None:
		_temp_root.destroy()
	return result['val']

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
	# Restore saved window layout: seed the video window props before it opens,
	# and remember the figure geometries to reapply once the figures exist.
	_layout = SWS_utils.load_layout()
	SWS_utils.restore_video_window_props(_layout)
	i = 0
	this_bin = 1*d['fsd']*d['epochlen'] #number of EEG data points in one epoch
	eeg_AD0 = np.load(os.path.join(d['savedir'],'AD0_downsampled', 'downsampEEG_Acq'+a+'_hr'+str(h)+'.npy'))
	eeg_AD2 = np.load(os.path.join(d['savedir'],'AD2_downsampled', 'downsampEEG_Acq'+a+'_hr'+str(h)+'.npy'))

	EEG_t = np.arange(np.size(eeg_AD0))/d['fsd'] #time array for EEG data
	start_trace = int(i-(4*d['epochlen'])) #timepoint in seconds that the plotted trace will start
	end_trace = int(i + (5*d['epochlen'])) #timepoint in seconds that the plotted trace will end

	if d['vid']:
		timestamp_df = pd.read_pickle(os.path.join(d['savedir'], 'All_timestamps.pkl'))
		try:
			this_timestamp = SWS_utils.pulling_timestamp(timestamp_df, acq_start, eeg_AD0, d['fsd'])
			print(f'Using timestamp: {this_timestamp}')
			cap, fps = SWS_utils.load_video(d, this_timestamp)
		except IndexError:
			d['vid'] = 0
			print("Timestamp information not available, turning off video access for this acquisition")

	print('loading the theta ratio...')
	ThD = SWS_utils.get_ThD(eeg_AD2, d['fsd'],
		cache_file = SWS_utils.thd_cache_path(d['savedir'], a, h, 2)) #array of ThD values per second
	ThD_t = np.arange(0, np.size(ThD))

	# fig2 rows: two +/-10 min overview spectrograms (ax6/ax7), zoomed velocity
	# (ax9) and EMG (ax10), and the clickable per-epoch state strip (ax_state).
	# The zoomed theta/delta trace (ax8) is drawn on a hidden throwaway axis so the
	# state strip can take its visible slot. ax6/ax7 must stay the first two axes
	# because SWS_utils.update_raw_trace indexes fig2.axes[0:2] for the overviews
	# and fig2.axes[2:5] for the zoomed velocity/EMG/state panels.
	fig2, (ax6, ax7, ax9, ax10, ax_state) = plt.subplots(nrows=5, ncols=1, figsize=(14, 8))
	ax8 = fig2.add_axes([0, 0, 1e-3, 1e-3])
	ax8.set_visible(False)
	fig1, ax1, ax2, ax3, ax4, ax5, state_img = SWS_utils.create_prediction_figure(d, State_input, is_predicted, clf,
		Features, d['fsd'], eeg_AD0, eeg_AD2, this_emg, EEG_t, d['epochlen'], start_trace, end_trace,
		d['Maximum_Frequency'], d['Minimum_Frequency'], [ax6, ax7], v = v, a = a, h = h)
	
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
		epochlen = d['epochlen'], ThD_ylims = [0,30], emg_ylims = ([-2, 2]), v_ylims = v_ylims)


	ax6.set_xlim([-600, 600])
	ax7.set_xlim([-600, 600])
	line4 = ax6.axvline(0, linewidth = 2, color = 'k')
	line5 = ax7.axvline(0, linewidth = 2, color = 'k')

	fig2.tight_layout()
	markers = SWS_utils.make_marker(fig1, this_bin/d['fsd'], d['epochlen'])


	plt.ion()
	State = deepcopy(State_input)

	# Crash recovery: if an autosave file is left over (e.g. the GUI crashed last
	# time before the user confirmed saving), offer to load it instead.
	recovery_path = _recovery_path(d, a, h)
	os.makedirs(os.path.dirname(recovery_path), exist_ok=True)
	if os.path.exists(recovery_path):
		if _ask_yes_no('Recover unsaved scoring',
				f'Unsaved scoring was found for Acq {a} hr {h} (possible earlier crash).\n'
				'Recover it? (No keeps the current states.)'):
			try:
				State = np.load(recovery_path)
				print('Recovered unsaved scoring from ' + recovery_path)
			except Exception as e:
				print(f'Could not load recovery file: {e}')
		else:
			try:
				os.remove(recovery_path)
			except OSError:
				pass

	# Track which epoch is centered in the detailed window (0 = first epoch) so the
	# state strip and the strip-click handler always refer to the right epochs.
	this_epoch_t = 0
	# Draw the per-epoch state strip for the initial window, and leave room at the
	# bottom for its x-label (tight_layout already ran before the strip existed).
	draw_state_strip(ax_state, State, this_epoch_t, start_trace, end_trace, d['epochlen'])
	fig2.subplots_adjust(bottom=0.1)
	fig2.canvas.draw()
	#init cursor and it's libraries from SW_Cursor.py
	# Pass all fig1 axes for full-height crosshair
	all_fig1_axes = [ax1, ax2, ax3, ax4, ax5]
	all_fig2_axes = [ax6, ax7, ax8, ax9, ax10]
	cursor = Cursor(ax1, ax2, ax5, all_axes=all_fig1_axes, epochlen=d['epochlen'], fig2_axes=all_fig2_axes)
	
	# Set up video data for preview mode
	if d['vid']:
		cursor.video_cap = cap
		cursor.video_timestamp = this_timestamp
		cursor.video_d = d
	
	# Set up magnify callback
	def magnify_update(time_sec):
		"""Update fig2 to show around cursor position."""
		half_window = cursor.magnify_half_window
		mag_start = time_sec - half_window
		mag_end = time_sec + half_window
		
		# Update zoomed traces (ax8, ax9, ax10)
		start_idx_ThD = np.where(long_ThD_t >= mag_start)[0]
		end_idx_ThD = np.where(long_ThD_t <= mag_end)[0]
		if len(start_idx_ThD) > 0 and len(end_idx_ThD) > 0:
			s_idx, e_idx = start_idx_ThD[0], end_idx_ThD[-1]
			line1.set_xdata(long_ThD_t[s_idx:e_idx+1])
			line1.set_ydata(long_ThD[s_idx:e_idx+1])
		
		if long_emg is not None:
			start_idx_emg = np.where(long_emg_t >= mag_start)[0]
			end_idx_emg = np.where(long_emg_t <= mag_end)[0]
			if len(start_idx_emg) > 0 and len(end_idx_emg) > 0:
				s_idx, e_idx = start_idx_emg[0], end_idx_emg[-1]
				line2.set_xdata(long_emg_t[s_idx:e_idx+1])
				line2.set_ydata(long_emg[s_idx:e_idx+1])
				if cursor.magnify_emg_ylim is not None:
					ax10.set_ylim(cursor.magnify_emg_ylim)
		
		if long_v is not None:
			v_idx = np.where(np.logical_and(long_v_t >= mag_start, long_v_t <= mag_end))[0]
			if len(v_idx) > 0:
				line3.set_xdata(long_v_t[v_idx])
				line3.set_ydata(long_v[v_idx])
		
		# Align ALL detailed axes to the same x range
		ax8.set_xlim(mag_start, mag_end)
		ax9.set_xlim(mag_start, mag_end)
		ax10.set_xlim(mag_start, mag_end)
		
		# Update additional spectrograms xlim
		ax6.set_xlim([time_sec - half_window, time_sec + half_window])
		ax7.set_xlim([time_sec - half_window, time_sec + half_window])
		line4.set_xdata([time_sec, time_sec])
		line5.set_xdata([time_sec, time_sec])
		
		# Invalidate fig2 background after magnify update
		cursor.background_fig2 = None
		fig2.canvas.draw_idle()
	
	cursor.magnify_callback = magnify_update

	def on_state_strip_click(event):
		"""Click an epoch in the detailed state strip to relabel just that epoch.

		Maps the click (x is seconds relative to the current epoch) to an absolute
		epoch index, asks for the new state via the same popup used elsewhere,
		applies it, autosaves to the recovery file, and refreshes both the main
		hypnogram image and the strip."""
		if event.inaxes is not ax_state or event.xdata is None or event.button != 1:
			return
		rel_epoch = int(math.floor(event.xdata / d['epochlen']))
		cur_idx = int(round(this_epoch_t / d['epochlen']))
		abs_idx = cur_idx + rel_epoch
		if abs_idx < 0 or abs_idx >= len(State):
			print('That epoch is outside the recording.')
			return
		popup_xy = cursor._get_screen_xy(event)
		new_state = choose_state_popup(popup_xy)
		if new_state is None:
			return
		State[abs_idx] = new_state
		np.save(recovery_path, State)
		SWS_utils.refresh_state_image(state_img, State)
		fig1.canvas.draw()
		cursor.background = fig1.canvas.copy_from_bbox(fig1.bbox)
		draw_state_strip(ax_state, State, this_epoch_t, start_trace, end_trace, d['epochlen'])
		fig2.canvas.draw()
		print(f'Epoch {abs_idx} set to state {new_state}.')

	# Connect fig1 events
	cID = fig1.canvas.mpl_connect('button_press_event', cursor.on_click)
	cID4 = fig1.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
	fig1.canvas.mpl_connect('resize_event', cursor.on_resize)

	# Connect fig2 events for cursor interaction, plus state-strip clicks.
	fig2.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move_fig2)
	fig2.canvas.mpl_connect('resize_event', cursor.on_resize_fig2)
	fig2.canvas.mpl_connect('key_press_event', cursor.on_press)
	fig2.canvas.mpl_connect('button_press_event', on_state_strip_click)

	#Ok so I think that the quotes is the specific event to trigger and the second arg is the function to run when that happens?
	cID2 = fig1.canvas.mpl_connect('axes_enter_event', cursor.in_axes)
	cID3 = fig1.canvas.mpl_connect('key_press_event', cursor.on_press)



	#This is the loop that manages the interface
	plt.show()

	# Reapply saved figure positions now that the windows exist.
	SWS_utils.apply_figure_geometry(fig1, _layout.get('fig1'))
	SWS_utils.apply_figure_geometry(fig2, _layout.get('fig2'))

	# Auto-open the video at the first frame (previously required a keypress).
	if d['vid'] and cursor.video_cap is not None:
		try:
			cursor._toggle_preview_window()
		except Exception as e:
			print(f'Could not auto-open video: {e}')

	DONE = False
	while not DONE:
		# Use short timeout so key events on either figure are processed
		try:
			plt.waitforbuttonpress(timeout=0.15)
		except Exception:
			plt.pause(0.05)
		# Skip iteration if no flags are set (timeout expired with no action)
		if not cursor.replot and not cursor.change_bins and not cursor.DONE:
			continue

		if cursor.replot:
			print("Replot of fig 1. called!")
			this_epoch_t = math.floor(cursor.replotx/d['epochlen'])*d['epochlen']
			replot_start = start_trace + this_epoch_t
			replot_end = end_trace + this_epoch_t
			print('Epoch Start Time = ' + str(this_epoch_t) + ' seconds')
			print('Start Trace = '+str(replot_start) + ' seconds')
			print('End Trace = ' + str(replot_end) + ' seconds')

			# Update current epoch marker FIRST
			cursor.current_epoch_t = this_epoch_t
			bin_idx = int(this_epoch_t // d['epochlen'])
			cursor.epoch_marker.set_xdata([bin_idx, bin_idx])
			fig1.canvas.draw_idle()
			fig1.canvas.flush_events()

			# Update fig2 to show this epoch (when not in magnify mode) BEFORE video
			if not cursor.magnify_mode and cursor.magnify_callback is not None:
				cursor.magnify_callback(this_epoch_t)
				fig2.canvas.flush_events()

			SWS_utils.update_raw_trace(fig1, fig2, line1, line2, line3, line4, line5, long_emg,
				long_emg_t, long_ThD, long_ThD_t, long_v, long_v_t, markers, this_epoch_t,
				replot_start, replot_end, d['epochlen'])

			# Redraw the per-epoch state strip for the newly-centered window
			# (update_raw_trace touched ax_state's xlim; this resets it).
			draw_state_strip(ax_state, State, this_epoch_t, start_trace, end_trace, d['epochlen'])
			fig2.canvas.draw_idle()

			if d['vid']:
				if this_epoch_t-d['epochlen'] < 0:
					print('No video available for this bin')
				else:
					# Play video snippet at this epoch (same as before)
					vid_start = int(this_timestamp.index[this_timestamp['Offset_Time']>(this_epoch_t-d['epochlen'])][0])
					vid_end = int(this_timestamp.index[this_timestamp['Offset_Time']<((this_epoch_t)+(d['epochlen']*2))][-1])
					SWS_utils.pull_up_movie(d, cap, vid_start, vid_end, 
						this_video, d['epochlen'], this_timestamp)


			plt.show()
			# Invalidate blitting background so new marker positions are captured
			cursor.background = None
			cursor.replot = False


			# Flip back the params

		if cursor.change_bins:
			bins = np.sort(cursor.bins)
			start_bin = int(bins[0])
			end_bin = int(bins[1])
			print(f'changing bins: {start_bin} to {end_bin}')

			# 'm' (microarousal) forces a single Wake bin and skips the popup.
			forced = getattr(cursor, 'forced_state', None)
			if forced is not None:
				new_state = forced
				cursor.forced_state = None
			else:
				new_state = choose_state_popup(cursor.popup_xy)
				if new_state is None:
					new_state = int(input('What state should these be?: '))

			# --- State edit (same array logic as before) ---
			State[start_bin:end_bin] = new_state
			if end_bin == len(State)-1:
				State[end_bin] = new_state
			# Autosave to the recovery file (not the canonical StatesAcq) so a crash
			# is recoverable but the real scoring isn't overwritten until the user
			# confirms saving on close.
			np.save(recovery_path, State)

			# --- Fast display update ---
			# Update the state image, redraw fig1 once to show the new colors, and
			# recapture the blit background from that same draw so the next mouse
			# move doesn't trigger a second full redraw.
			SWS_utils.refresh_state_image(state_img, State)
			fig1.canvas.draw()
			cursor.background = fig1.canvas.copy_from_bbox(fig1.bbox)
			# Reflect the edit in the detailed state strip too.
			draw_state_strip(ax_state, State, this_epoch_t, start_trace, end_trace, d['epochlen'])
			fig2.canvas.draw_idle()
			cursor.bins = []
			cursor.clicked = False
			cursor.change_bins = False
		if cursor.DONE:
			DONE = True

	print('successfully left GUI')

	# Ask whether to save. 'Yes' writes the canonical StatesAcq file; 'No' leaves
	# the previously-saved states untouched. Either way the recovery autosave is
	# cleared, since this is a clean exit (recovery only matters after a crash).
	states_path = os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy')
	save_it = _ask_yes_no('Save scoring',
		f'Save sleep states for Acq {a} hr {h}?')
	if save_it:
		np.save(states_path, State)
		print('Saved states to ' + states_path)
	else:
		print('Not saving; existing StatesAcq file (if any) left unchanged.')
	try:
		if os.path.exists(recovery_path):
			os.remove(recovery_path)
	except OSError:
		pass

	# Persist window layout (figure geometry + video window) for the next launch.
	try:
		if getattr(cursor, 'preview_window_open', False) and getattr(cursor, '_preview_visible', False):
			cursor._update_preview_window_props()
	except Exception:
		pass
	SWS_utils.update_layout(
		fig1=SWS_utils.get_figure_geometry(fig1),
		fig2=SWS_utils.get_figure_geometry(fig2),
		video=dict(SWS_utils._video_window_props),
	)
	# Release any lazily-opened video captures so re-launching another
	# acquisition in the same process (via the launcher) doesn't leak handles.
	try:
		if d['vid'] and 'cap' in locals() and hasattr(cap, 'release_all'):
			cap.release_all()
	except Exception:
		pass
	cv2.destroyAllWindows()
	plt.close('all')

	return State


def scored_acquisitions(d):
	"""Return the sorted list of acquisition numbers that already have a State file."""
	state_files = glob.glob(os.path.join(d['savedir'], 'StatesAcq*.npy'))
	scored = []
	for sf in state_files:
		filename = os.path.split(sf)[1]
		idx1 = filename.find('q')
		idx2 = filename.find('_')
		try:
			scored.append(int(filename[idx1+1:idx2]))
		except ValueError:
			continue
	return sorted(set(scored))

def score_acquisition(d, a, use_model=True, mode='s',
		update_model_after=None, update_log_after=None):
	"""Score (or check) a single acquisition without terminal prompts.

	Parameters that used to be answered via input() are now explicit so the
	launcher can drive this directly:
	  - a: acquisition number (int or str)
	  - use_model: use the random-forest model to pre-predict (mode 's' only)
	  - mode: 's' = score new dataset, 'c' = check/fix existing scoring
	  - update_model_after / update_log_after: True/False to skip the prompt,
	    or None to fall back to the interactive prompt (legacy behavior).
	"""
	a = str(a)
	warnings.filterwarnings("ignore")
	print('Loading EEG and EMG....')
	downsampEEG = np.load(os.path.join(d['savedir'],'downsampEEG_Acq'+str(a)+'.npy'))
	if d['emg']:
		downsampEMG = np.load(os.path.join(d['savedir'],'downsampEMG_Acq'+str(a)+'.npy'))
	acq_len = np.size(downsampEEG)/d['fsd'] # fs: sampling rate, fsd: downsampled sampling rate
	hour_segs = math.ceil(acq_len/3600) # acq_len in seconds, convert to hours
	print('This acquisition has ' +str(hour_segs)+ ' segments.')

	acq_start = SWS_utils.get_AcqStart(d, a, acq_len)
	print(f'Acquisition {a} start time: {acq_start} Acquisition length (s): {acq_len}')

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
		# Cached feature extraction (~15s -> ~0.1s on a warm cache).
		FeatureDict = SWS_utils.build_feature_dict_cached(d, a, h, eeg_df, normVal)
		print(f'Acquisition start time: {acq_start}')
		this_video, v, this_motion = SWS_utils.initialize_vid_and_move(d, a, acq_start, acq_len)
		print(f'Video name: {this_video}')
		if d['movement']:
			FeatureDict['Velocity'] = v[0]
		FeatureDict['animal_name'] = np.full(len(FeatureDict[list(FeatureDict.keys())[0]]), d['mouse_name'])

		os.chdir(d['savedir'])
		this_emg = eeg_df['EMG']
		State = None
		if mode == 'c':
			try:
				# if some portion of the file has been previously scored
				State = np.load(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'))
				wrong, = np.where(np.isnan(State))
				State[wrong] = 0
				s, = np.where(State == 0)

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

		else:  # mode == 's'
			if use_model:
				jobname = SWS_utils.build_joblib_name(d)
				try:
					clf = joblib.load(os.path.join(d['model_dir'], jobname))
				except FileNotFoundError:
					print("You don't have a model to work with.")
					return

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

		if State is None:
			# Nothing was scored (e.g. 'check' mode with no existing file).
			plt.close('all')
			continue

		FeatureDict['State'] = State

		if update_model_after is None:
			update = input('Do you want to update the model?: y/n ') == 'y'
		else:
			update = bool(update_model_after)
		if update:
			update_model(d, FeatureDict)
			model_log(d['modellog_dir'], 0, d['species'], d['mouse_name'], d['mod_name'], a)

		if update_log_after is None:
			logq = input('Do you want to update your personal log?: y/n ') == 'y'
		else:
			logq = bool(update_log_after)
		if logq:
			personal_log(d['personallog_dir'], d['mouse_name'], d['savedir'], a)

		plt.close('all')
			# Store the result.

def start_swscoring(d):
	"""Legacy terminal-prompt entry point. The launcher is now the primary UI;
	this remains as a fallback and delegates to score_acquisition."""
	# mostly for deprecated packages
	print('this code is supressing warnings')
	warnings.filterwarnings("ignore")
	print('These are the available acquisitions: '+ str(d['Acquisition']))
	print('These are the acquisitions that have a previous State file: ' + str(scored_acquisitions(d)))
	a = input('Which acqusition do you want to score?')
	check = input('Do you want to check and fix existing scoring (c) or score new dataset (s)?: c/s ')
	while check != 'c' and check != 's':
		check = input(
			'Only c/s is accepted. Do you want to check and fix existing scoring (c) or score new dataset (s)?: c/s ')
	use_model = True
	if check == 's':
		use_model = input('Use a random forest? y/n: ') == 'y'
	# update_model_after / update_log_after left as None -> prompt (legacy).
	score_acquisition(d, a, use_model=use_model, mode=check)

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
	# The central launcher is now the entry point. An optional settings-JSON path
	# pre-loads that file; otherwise the launcher opens with a Browse button.
	if len(args) > 2:
		print("You only need to specify the path of your Score_Settings.json (optional). For instance, run `python New_SWS.py /home/ChenLab_Sleep_Scoring/Score_Settings.json`.")
	else:
		settings = args[1] if len(args) == 2 else None
		from neuroscience_sleep_scoring.ScoringLauncher import launch
		launch(settings)
