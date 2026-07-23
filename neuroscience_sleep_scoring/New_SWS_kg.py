import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.widgets import Button
import glob
from copy import deepcopy
import sys
import select
import os
import math
import json
import cv2
import joblib
import pandas as pd
import warnings
from neuroscience_sleep_scoring import SWS_utils, extract_data
from neuroscience_sleep_scoring.SW_Cursor import Cursor
from datetime import datetime


# Color used for each sleep-state code in the GUI, matching SWS_utils.correct_bins:
#   1 = green, 2 = blue, 3 = red, 4 = purple; anything else (0/unscored/NaN) = white.
STATE_COLORS = {1: 'green', 2: 'blue', 3: 'red', 4: 'purple'}


def state_to_color(state_value):
	"""Return the plotting color for a single sleep-state code.

	Mirrors the color scheme used in SWS_utils.correct_bins so the zoomed state
	strip matches the main scoring figure. Unscored epochs (0) and NaNs map to white.
	"""
	try:
		if np.isnan(state_value):
			return 'white'
	except TypeError:
		pass
	return STATE_COLORS.get(int(state_value), 'white')


def draw_state_strip(ax_state, State, this_epoch_t, start_trace, end_trace, epochlen, selected=None):
	"""Draw the sleep state of every epoch visible in the zoomed window.

	The zoomed trace axes use a coordinate system relative to the current epoch: the
	current epoch starts at x = 0 and each epoch spans `epochlen` seconds, so an epoch
	`rel` positions away from the current one is drawn at x = rel*epochlen. This fills
	`ax_state` with one colored rectangle per visible epoch (color taken from `State`),
	outlines the current epoch in yellow to match the trace highlight, and keeps the
	strip aligned with the traces above it.

	`selected`, if given, is a set of absolute epoch indices picked (by clicking the strip)
	for a pending bulk relabel; those epochs are drawn white regardless of their stored
	state so the selection stays visible until a new state is typed in the terminal.

	Called once when the figure is built and again after every replot, selection, or state
	edit, so the strip always reflects the latest contents of the live `State` array.
	"""
	ax_state.clear()
	# Absolute index of the epoch currently centered in the window.
	cur_idx = int(round(this_epoch_t / epochlen))
	# Range of epoch offsets needed to cover the visible window [start_trace, end_trace].
	first_rel = int(math.floor(start_trace / epochlen))
	last_rel = int(math.ceil(end_trace / epochlen))
	for rel in range(first_rel, last_rel):
		abs_idx = cur_idx + rel
		x = rel * epochlen
		if selected and abs_idx in selected:
			color = 'white'  # picked for a pending relabel
		elif 0 <= abs_idx < len(State):
			color = state_to_color(State[abs_idx])
		else:
			color = 'white'  # epochs outside the recording
		ax_state.add_patch(patch.Rectangle((x, 0), epochlen, 1, color=color, ec='k', lw=0.5))
	# Outline the current epoch (x = 0..epochlen) to match the yellow marker on the traces.
	ax_state.add_patch(patch.Rectangle((0, 0), epochlen, 1, fill=False, ec='#fac205', lw=2.5))
	ax_state.set_xlim([start_trace, end_trace])
	ax_state.set_ylim([0, 1])
	ax_state.set_yticks([])
	ax_state.set_ylabel('Sleep\nState')
	ax_state.set_xlabel('Time (s) relative to current epoch')


class BlitCrosshair:
	"""Fast vertical/horizontal cursor shared across the main figure's stacked panels.

	SW_Cursor.Cursor only draws a crosshair on the states axis and forces a full canvas
	redraw on every mouse move, which is slow. This replacement draws a vertical line on
	every panel (both spectrograms, the states strip, velocity, and EMG) plus a horizontal
	line on whichever panel the cursor is over, and uses blitting so the lines track the
	cursor smoothly.

	The states axis is plotted in epoch units while the spectrogram/velocity/EMG axes are
	in seconds; the hovered x-position is converted to seconds once and then mapped back
	into each axis's own units, so the vertical line marks the same instant on every panel.
	"""

	def __init__(self, fig, second_axes, epoch_ax, epochlen):
		self.fig = fig
		self.canvas = fig.canvas
		self.epoch_ax = epoch_ax
		self.epochlen = epochlen
		self.background = None
		self.axes = list(second_axes) + [epoch_ax]
		# Animated artists are skipped by normal draws and rendered by hand during a blit.
		self.vlines = {ax: ax.axvline(ax.get_xlim()[0], color='k', lw=0.8, animated=True)
			for ax in self.axes}
		self.hlines = {ax: ax.axhline(ax.get_ylim()[0], color='k', lw=0.8, ls='--', animated=True)
			for ax in self.axes}
		# Recapture the clean background whenever the figure is fully redrawn (e.g. a replot).
		self.canvas.mpl_connect('draw_event', self._on_draw)

	def _on_draw(self, event):
		"""Cache the figure (minus the animated crosshair) as a bitmap for fast restores."""
		self.background = self.canvas.copy_from_bbox(self.fig.bbox)

	def on_move(self, event):
		"""Redraw the crosshair at the cursor position using blitting (no full redraw)."""
		# We need a cached background before we can blit; grab one on the first move.
		if self.background is None:
			self.background = self.canvas.copy_from_bbox(self.fig.bbox)
		# Erase the previous crosshair by restoring the clean background.
		self.canvas.restore_region(self.background)
		if event.inaxes in self.vlines:
			# Convert the hovered x into seconds, then place a vertical line on every panel.
			if event.inaxes is self.epoch_ax:
				t_sec = event.xdata * self.epochlen
			else:
				t_sec = event.xdata
			for ax, vl in self.vlines.items():
				x = t_sec / self.epochlen if ax is self.epoch_ax else t_sec
				vl.set_xdata([x, x])
				ax.draw_artist(vl)
			# Horizontal line only on the panel actually under the cursor.
			hl = self.hlines[event.inaxes]
			hl.set_ydata([event.ydata, event.ydata])
			event.inaxes.draw_artist(hl)
		self.canvas.blit(self.fig.bbox)


# ---------------------------------------------------------------------------
# How-To help page (opened by the '?' button at the bottom of the main figure)
# ---------------------------------------------------------------------------
# Left column: what each plot shows.
HOWTO_PLOTS = """MAIN FIGURE  (top to bottom)
  1. Spectrogram - EEG channel 1
       Time-frequency power of the frontal EEG.
       Brighter = more power.  x = time, y = freq (Hz).
  2. Predicted States (hypnogram)
       One colored bar per 4 s epoch = its sleep state.
       Black trace = model confidence (higher = most certain).
  3. Spectrogram - EEG channel 2
       Same as #1 for the hippocampal EEG channel.
  4. Velocity
       Animal movement speed in pixels/s.
  5. EMG amplitude
       Muscle activity; higher amp. during wakefulness.

ZOOMED FIGURE  (second window)
  - Top 2 panels: +/-10 min overview spectrograms;
       the black line marks the current epoch.
  - Velocity (zoom) and EMG (zoom): window centered on
       the current epoch (x = 0), epoch highlighted yellow.
  - Sleep State strip (bottom): state of every epoch in
       the window; current epoch outlined in yellow."""

# Right column: every click / button / key.
HOWTO_CONTROLS = """MOUSE
  - Click top spectrogram (main figure):
      Center the zoomed window there (and play that
      Epoch's video with window before and after).
  - Click the Predicted States panel twice:
      First click will be first changed epoch
      Second click will be first unchanged epoch
      Then type the new state in the terminal.
  - Click epochs in the zoomed State strip:
      Toggle-select them (they turn white); then type
      one state number in the terminal to set them all.

KEYS  (figure window must have focus; use lowercase)
  - Left / Right arrow: move 1 epoch (4 s); no video.
  - v: stops video; with no video, it plays the current 
       Epoch's video on repeat
       Press v again to stop.
  - l: hover a plot, press l to place a vertical
       Reference line across the panels; l again removes it.
  - d: done - close the GUI, prompt to save / update.

BUTTONS
  - ?  (this page): open the How-To help.
  - Save (toolbar disk icon): save a figure snapshot."""

# Sleep-state score number -> the color it is drawn in.
HOWTO_STATES = [
	('Wake (1)', 'green'),
	('NREM sleep (2)', 'blue'),
	('REM sleep (3)', 'red'),
]

# Biology reference table (matches the standard scoring criteria).
HOWTO_TABLE_COLS = ['', 'NREM sleep', 'REM sleep', 'Wakefulness']
HOWTO_TABLE_ROWS = [
	['EEG amplitude', 'High', 'Low', 'Low'],
	['Dominant EEG frequency', 'Delta band (0.5-4 Hz)', 'Theta band (6-10 Hz)', 'None'],
	['EMG amplitude', 'Low', 'Low', 'High'],
]


def show_how_to(event=None):
	"""Open (or refresh) the How-To window for the scoring GUI.

	Lays out, in a standalone figure: a description of every plot (main and zoomed
	figures), every mouse click / button / key, a color legend for the sleep-state score
	numbers, and the reference table of how each state looks in the EEG/EMG. Bound to the
	'?' button, so `event` is the Matplotlib button event (ignored).
	"""
	# Open at roughly the size the page is comfortably read at (~925 x 610 px at 100 dpi).
	help_fig = plt.figure('How To - Sleep/Wake Scoring', figsize=(9.25, 7))
	help_fig.clf()  # rebuild cleanly if the window was opened before
	help_fig.suptitle('How To  -  Sleep / Wake Scoring GUI', fontsize=15, fontweight='bold')

	# Two text columns: plots on the left, controls on the right.
	help_fig.text(0.03, 0.93, HOWTO_PLOTS, va='top', ha='left', fontsize=8.5, family='monospace')
	help_fig.text(0.52, 0.93, HOWTO_CONTROLS, va='top', ha='left', fontsize=8.5, family='monospace')

	# Color legend for the sleep-state score numbers.
	leg_ax = help_fig.add_axes([0.04, 0.30, 0.92, 0.07])
	leg_ax.axis('off')
	leg_ax.set_xlim(0, len(HOWTO_STATES))
	leg_ax.set_ylim(0, 1)
	leg_ax.set_title('Sleep states  (score number -> color)', fontsize=10, loc='left')
	for i, (label, color) in enumerate(HOWTO_STATES):
		leg_ax.add_patch(patch.Rectangle((i + 0.03, 0.45), 0.2, 0.5, color=color, ec='k'))
		leg_ax.text(i + 0.03, 0.30, label, fontsize=8, va='top')

	# Reference table: what each state looks like in the data (from the scoring criteria).
	table_ax = help_fig.add_axes([0.05, 0.04, 0.9, 0.2])
	table_ax.axis('off')
	table_ax.set_title('What each state looks like in the data', fontsize=10, loc='left')
	tbl = table_ax.table(cellText=HOWTO_TABLE_ROWS, colLabels=HOWTO_TABLE_COLS,
		loc='center', cellLoc='center')
	tbl.auto_set_font_size(False)
	tbl.set_fontsize(9)
	tbl.scale(1, 1.7)
	for c in range(len(HOWTO_TABLE_COLS)):
		tbl[0, c].set_text_props(fontweight='bold')  # bold the header row

	help_fig.show()


def update_model(d, FeatureDict):
	"""Append a freshly-scored acquisition to the training set and retrain the model.

	`d` is the settings dictionary loaded from Score_Settings.json and `FeatureDict`
	holds the per-epoch features (delta/theta power, EMG variance, velocity, etc.)
	plus the human-corrected `State` labels for one acquisition.

	Steps:
	  1. If movement tracking is enabled, collapse the per-second velocity trace down
	     to one value per epoch so it lines up with the other features.
	  2. Replace any NaN EMG-variance values with 0 so the classifier can use them.
	  3. Append these rows to the on-disk training dataframe for this model.
	  4. Drop any leftover 'nan' EMG rows and retrain the random forest, saving it
	     under the appropriate joblib name (suffixed '_2chan' for two-EEG-channel setups).
	"""
	# Collapse the per-second velocity into one value per epoch (only if movement
	# tracking is on). Previously this was followed by a second, malformed call that
	# crashed; the single guarded call below is the correct behavior.
	if d['movement']:
		FeatureDict = SWS_utils.adjust_movement(FeatureDict, epochlen=d['epochlen'])

	# The classifier can't train on NaNs, so zero out any missing EMG-variance values.
	if 'EMGvar' in FeatureDict.keys():
		FeatureDict['EMGvar'][np.isnan(FeatureDict['EMGvar'])] = 0

	# Turn this acquisition's features into a dataframe of new training rows.
	df_additions = pd.DataFrame(FeatureDict)

	# Two-channel EEG recordings train a separate model, so tag the model name.
	mod_name = d['mod_name']
	if len(d['EEG channel']) == 2:
		mod_name = mod_name + '_2chan'

	# Append the new rows to the accumulated training dataframe on disk.
	Sleep_Model = SWS_utils.update_sleep_df(d['model_dir'], mod_name, df_additions)
	jobname = SWS_utils.build_joblib_name(d)
	x_features = SWS_utils.get_xfeatures(FeatureDict)

	# Remove any rows whose EMG variance is the string 'nan' before training.
	if 'EMGvar' in Sleep_Model.columns:
		Sleep_Model = Sleep_Model.drop(index=np.where(Sleep_Model['EMGvar'].isin(['nan']))[0])

	# Retrain and persist the random-forest classifier with the expanded dataset.
	SWS_utils.retrain_model(Sleep_Model, x_features, d['model_dir'], jobname)


def display_and_fix_scoring(d, a, h, this_emg, State_input, is_predicted, clf, Features, this_video,
	acq_start, v=None, movement_df=None, buffer=4):
	"""Open the interactive scoring GUI for one hour-long segment and return the edited states.

	Parameters:
	  d            - settings dictionary
	  a, h         - acquisition number and hour index (identify which segment to load)
	  this_emg     - EMG trace for this segment
	  State_input  - starting per-epoch state array (model prediction or prior scoring)
	  is_predicted - True if State_input came from the model (changes how the figure is built)
	  clf          - the trained classifier (used by the prediction figure), or None
	  Features     - per-epoch feature matrix used to draw the prediction figure
	  this_video   - path to the behavior video for this segment
	  acq_start    - wall-clock start time of the acquisition (for aligning video timestamps)
	  v            - (velocity, time) arrays from movement tracking, or None
	  movement_df  - movement dataframe (currently unused here, kept for call compatibility)
	  buffer       - number of epochs of padding added to each end of the zoomed traces

	The user clicks/keys to reassign sleep states; edits are written to State and saved to
	StatesAcq{a}_hr{h}.npy. The function blocks until the user signals DONE, then returns State.
	"""
	plt.ion()
	i = 0
	this_bin = 1 * d['fsd'] * d['epochlen']  # number of EEG data points in one epoch

	# Load the two downsampled EEG channels for this acquisition/hour.
	eeg_AD0 = np.load(os.path.join(d['savedir'], 'AD0_downsampled',
		'downsampEEG_Acq' + a + '_hr' + str(h) + '.npy'))
	eeg_AD2 = np.load(os.path.join(d['savedir'], 'AD2_downsampled',
		'downsampEEG_Acq' + a + '_hr' + str(h) + '.npy'))

	EEG_t = np.arange(np.size(eeg_AD0)) / d['fsd']  # time array (seconds) for the EEG data
	start_trace = int(i - (4 * d['epochlen']))  # start time (s) of the initially plotted raw trace
	end_trace = int(i + (5 * d['epochlen']))    # end time (s) of the initially plotted raw trace

	# If video review is enabled, line the video up with the EEG using the timestamp table.
	# If that table is missing we simply disable video for this acquisition.
	if d['vid']:
		timestamp_df = pd.read_pickle(os.path.join(d['savedir'], 'All_timestamps.pkl'))
		try:
			this_timestamp = SWS_utils.pulling_timestamp(timestamp_df, acq_start, eeg_AD0, d['fsd'])
			cap, fps = SWS_utils.load_video(d, this_timestamp)
		except IndexError:
			d['vid'] = 0
			print("Timestamp information not available, turning off video access for this acquisition")

	# Theta/delta ratio is a key discriminator for REM vs non-REM; compute it per second.
	print('loading the theta ratio...')
	ThD = SWS_utils.get_ThD(eeg_AD2, d['fsd'])  # array of theta/delta ratio values, one per second
	ThD_t = np.arange(0, np.size(ThD))

	# fig2 holds the overview/zoomed axes plus a per-epoch state strip; fig1 holds the
	# main scoring panels. ax6/ax7 = +/-10 min overview, ax9/ax10 = zoomed velocity and
	# EMG, and ax_state = the sleep-state strip for the window.
	# ax6 and ax7 must stay the first two axes since update_raw_trace indexes fig2.axes[0:2].
	fig2, (ax6, ax7, ax9, ax10, ax_state) = plt.subplots(nrows=5, ncols=1, figsize=(14, 8))
	# create_zoomed_fig / update_raw_trace (in SWS_utils, which we don't edit) always draw
	# and update a theta/delta trace on their first axis. We no longer want that trace shown,
	# so hand them a hidden throwaway axis to draw it on, keeping the visible panels limited
	# to velocity, EMG, and the state strip.
	ax8 = fig2.add_axes([0, 0, 1e-3, 1e-3])
	ax8.set_visible(False)
	fig1, ax1, ax2, ax3, ax4, ax5 = SWS_utils.create_prediction_figure(d, State_input, is_predicted, clf,
		Features, d['fsd'], eeg_AD0, eeg_AD2, this_emg, EEG_t, d['epochlen'], start_trace, end_trace,
		d['Maximum_Frequency'], d['Minimum_Frequency'], [ax6, ax7], v=v)

	# Capture the auto-scaled y-limits so the zoomed velocity/EMG plots match the main ones.
	v_ylims = list(ax4.get_ylim())
	emg_ylims = list(ax5.get_ylim())

	# Pad the traces on both ends so the very first and last epochs can still be centered.
	buffer_seconds = buffer * d['epochlen']  # padding (s) added to the start and end of each trace
	long_ThD, long_ThD_t = SWS_utils.add_buffer(ThD, ThD_t, buffer_seconds, fs=1)
	long_emg, long_emg_t = SWS_utils.add_buffer(this_emg, EEG_t, buffer_seconds, fs=200)
	if d['movement']:
		long_v, long_v_t = SWS_utils.add_buffer(np.insert(v[0], 0, 0), np.insert(v[1], 0, 0),
			buffer_seconds, fs=1 / int(d['epochlen']))
	else:
		long_v = None
		long_v_t = None

	# Draw the three zoomed traces (EMG, theta/delta, velocity) and keep handles for updating.
	line1, line2, line3 = SWS_utils.create_zoomed_fig(ax8, ax9, ax10, long_emg, long_emg_t,
		long_ThD, long_ThD_t, long_v, long_v_t, start_trace, end_trace,
		epochlen=d['epochlen'], ThD_ylims=[0, 30], emg_ylims=([-0.25, 0.25]), v_ylims=v_ylims)

	# The two overview axes span +/-10 min; the vertical line marks the current epoch.
	ax6.set_xlim([-600, 600])
	ax7.set_xlim([-600, 600])
	line4 = ax6.axvline(0, linewidth=2, color='k')
	line5 = ax7.axvline(0, linewidth=2, color='k')

	fig2.tight_layout()
	# Place the epoch markers at the initial current epoch (t = 0). Passing this_bin/fsd
	# (= epochlen) put the very first marker one epoch to the right of the current epoch,
	# so on the states panel it sat a sliver left of the true epoch boundary until the first
	# replot moved it. update_raw_trace keeps it on the current epoch after that.
	markers = SWS_utils.make_marker(fig1, 0, d['epochlen'])

	# Add a small, square '?' help button in the bottom-right corner of the main figure
	# (near the toolbar's save button) that opens the How-To page. This is created AFTER
	# make_marker so the per-axis epoch marker line isn't drawn through the button. Free a
	# little space at the bottom first so it doesn't sit on the EMG panel; keep a reference
	# so the Button isn't garbage-collected.
	fig1.subplots_adjust(bottom=0.1)
	help_ax = fig1.add_axes([0.955, 0.005, 0.027, 0.05])  # ~square on the 11x6 figure
	help_button = Button(help_ax, '?', color='#d9e6d9', hovercolor='#b8d4b8')
	help_button.label.set_fontsize(12)
	help_button.label.set_fontweight('bold')
	help_button.on_clicked(show_how_to)

	plt.ion()
	State = deepcopy(State_input)  # work on a copy so we don't mutate the caller's array

	# Track which epoch is centered in the zoomed window (0 = first epoch on open) so the
	# state strip can be redrawn for the right window after edits even before any replot.
	this_epoch_t = 0
	# Epochs picked (by clicking the strip) for a pending bulk relabel, held as absolute
	# epoch indices. They are applied all at once when the user types a state number in the
	# terminal; clicking a selected epoch again removes it from the selection.
	strip_selection = set()
	# Draw the per-epoch sleep states for the initial window (centered on epoch 0).
	draw_state_strip(ax_state, State, this_epoch_t, start_trace, end_trace, d['epochlen'],
		selected=strip_selection)
	# tight_layout ran before the strip's x-label existed, so leave room at the bottom now
	# to keep that label from being clipped (axes positions persist across strip redraws).
	fig2.subplots_adjust(bottom=0.1)
	fig2.canvas.draw()

	# The Cursor object (from SW_Cursor.py) tracks clicks/keypresses and which epochs are selected.
	cursor = Cursor(ax1, ax2, ax5)

	# Hide the Cursor's built-in crosshair: it lives only on the states axis and forces a
	# full, slow canvas redraw on every move. We replace it with a blitted crosshair below.
	cursor.horizontal_line.set_visible(False)
	cursor.vertical_line.set_visible(False)
	cursor.text.set_visible(False)

	# Fast crosshair shared across both spectrograms, the states strip, velocity, and EMG.
	# ax2 is in epoch units; ax1/ax3/ax4/ax5 are in seconds (handled inside BlitCrosshair).
	crosshair = BlitCrosshair(fig1, second_axes=[ax1, ax3, ax4, ax5], epoch_ax=ax2,
		epochlen=d['epochlen'])

	# Flag so arrow-key navigation can suppress the video pop-up: set in on_arrow_key and
	# cleared after each replot. Ordinary spectrogram clicks leave it False.
	arrow_nav = [False]

	# State for the 'l' reference line: the placed Line2D handles and whether it's showing.
	line_toggle = {'lines': [], 'on': False}

	# Matplotlib binds several keys by default; drop the ones we reuse so they don't also
	# trigger the built-in behavior: left/right (view-history nav) and 'l' (toggle the
	# hovered axis to log scale, which is what was flipping the spectrogram to log).
	plt.rcParams['keymap.back'] = [k for k in plt.rcParams['keymap.back'] if k != 'left']
	plt.rcParams['keymap.forward'] = [k for k in plt.rcParams['keymap.forward'] if k != 'right']
	plt.rcParams['keymap.yscale'] = [k for k in plt.rcParams['keymap.yscale'] if k != 'l']

	def on_arrow_key(event):
		"""Step the zoomed window one epoch left/right when an arrow key is pressed.

		Sets the cursor's replot target to the neighboring epoch and raises the replot
		flag, so the main event loop redraws the traces and state strip via the same path
		used for a spectrogram click. `arrow_nav` is set so that path skips the video
		pop-up. The target is clamped so we can't scroll past the start or end of the segment.
		"""
		if event.key == 'left':
			new_epoch_t = this_epoch_t - d['epochlen']
		elif event.key == 'right':
			new_epoch_t = this_epoch_t + d['epochlen']
		else:
			return
		max_epoch_t = (len(State) - 1) * d['epochlen']
		cursor.replotx = max(0, min(new_epoch_t, max_epoch_t))
		arrow_nav[0] = True
		cursor.replot = True

	def on_state_strip_click(event):
		"""Click epochs in the zoomed state strip to pick them for a bulk relabel.

		Each left-click toggles the clicked epoch in/out of the pending selection and paints
		the selected epochs white. Nothing is committed here: the selection is applied to all
		picked epochs at once by the main loop when a state number is typed in the terminal.
		The strip x-axis is in seconds relative to the current epoch, so the click position is
		converted to an absolute epoch index.
		"""
		# Only respond to a left-click landing inside the state strip.
		if event.inaxes is not ax_state or event.xdata is None or event.button != 1:
			return
		# Map the clicked x (seconds relative to current epoch) to an absolute epoch index.
		rel_epoch = int(math.floor(event.xdata / d['epochlen']))
		cur_idx = int(round(this_epoch_t / d['epochlen']))
		abs_idx = cur_idx + rel_epoch
		if abs_idx < 0 or abs_idx >= len(State):
			print('That epoch is outside the recording; nothing to select.')
			return

		# Toggle this epoch in/out of the pending selection.
		if abs_idx in strip_selection:
			strip_selection.discard(abs_idx)
			print(f'Deselected epoch {abs_idx}. {len(strip_selection)} epoch(s) selected.')
		else:
			strip_selection.add(abs_idx)
			print(f'Selected epoch {abs_idx}. {len(strip_selection)} epoch(s) selected. '
				'Type a state number in the terminal and press Enter to apply to all of them.')

		# Repaint the strip so the selected epochs show as white.
		draw_state_strip(ax_state, State, this_epoch_t, start_trace, end_trace, d['epochlen'],
			selected=strip_selection)
		fig2.canvas.draw()
		fig2.canvas.flush_events()

	def play_current_epoch_video():
		"""Play only the current epoch's behavior video, looping until 'v' is pressed.

		Triggered by pressing 'v' when no video window is open. Unlike the click/replot
		video (which also shows padding before and after the epoch), this plays just the
		frames of the epoch currently centered in the zoomed window, aligned to the same
		time range the main video marks as the SCORE WINDOW: from the first frame after
		this_epoch_t to the first frame after this_epoch_t + epochlen. The clip loops so the
		short epoch is watchable; pressing 'v' again while it plays stops it (matching the
		main video's behavior).
		"""
		if not d['vid']:
			print('Video review is off for this acquisition.')
			return
		epochlen = d['epochlen']
		# Same alignment pull_up_movie uses for its SCORE WINDOW (the epoch being scored).
		try:
			frame_start = int(this_timestamp.index[this_timestamp['Offset_Time'] > this_epoch_t][0])
			frame_end = int(this_timestamp.index[this_timestamp['Offset_Time'] > this_epoch_t + epochlen][0])
		except IndexError:
			print('No video available for this epoch.')
			return
		if frame_end <= frame_start:
			print('No video frames for this epoch.')
			return
		vid_fn = SWS_utils.get_videofn_from_csv(d, this_timestamp['Filename'][frame_start])
		if not cap[vid_fn].isOpened():
			print("Error opening video stream or file")
			return
		print(f'Playing current epoch ({this_epoch_t}-{this_epoch_t + epochlen} s). Press v to stop.')
		stop = False
		while not stop:
			shown = 0
			for f in np.arange(frame_start, frame_end):
				cap[vid_fn].set(1, f)
				ret, frame = cap[vid_fn].read()
				if ret:
					shown += 1
					cv2.imshow('Frame', frame)
					# waitKey(1) matches the main video's playback speed; 'v' stops it.
					if (cv2.waitKey(1) & 0xFF) == ord('v'):
						stop = True
						break
			if shown == 0:
				print('Could not read video frames for this epoch.')
				break
		cv2.destroyAllWindows()

	def on_video_key(event):
		"""Play the current epoch's video when 'v' is pressed and no video is already open.

		While a video is playing the cv2 window owns the 'v' key (to stop playback), so this
		matplotlib handler only fires when no clip is up, giving 'v' the requested dual role.
		"""
		if event.key == 'v':
			play_current_epoch_video()

	def on_key(event):
		"""Handle 'd' (done) and 'l' (toggle a placed vertical reference line).

		This replaces Cursor.on_press, whose 'l' branch is broken (its `self.lines` list is
		never initialized, so pressing 'l' raised an error and did nothing). 'd' ends scoring
		as before. 'l' places a dashed vertical line at the current mouse position across all
		main-figure panels and leaves it there; pressing 'l' again removes it. The hovered x is
		converted to seconds once and mapped into each panel's units (the states panel is in
		epochs, the rest in seconds) so the line marks the same instant on every plot.
		"""
		if event.key == 'd':
			print('DONE SCORING')
			cursor.DONE = True
		elif event.key == 'l':
			if line_toggle['on']:
				# Toggle the line off.
				for ln in line_toggle['lines']:
					ln.remove()
				line_toggle['lines'] = []
				line_toggle['on'] = False
				fig1.canvas.draw()
			else:
				# Toggle on: place it where the mouse currently is (must be over a panel).
				line_axes = [ax1, ax2, ax3, ax4, ax5]
				if event.inaxes not in line_axes or event.xdata is None:
					print("Hover the mouse over one of the plots, then press 'l' to place the line.")
					return
				t_sec = event.xdata * d['epochlen'] if event.inaxes is ax2 else event.xdata
				for ax in line_axes:
					x = t_sec / d['epochlen'] if ax is ax2 else t_sec
					line_toggle['lines'].append(ax.axvline(x, color='k', lw=1, ls='--'))
				line_toggle['on'] = True
				fig1.canvas.draw()

	# Wire up mouse and keyboard events on the main figure.
	cID = fig1.canvas.mpl_connect('button_press_event', cursor.on_click)
	cID4 = fig1.canvas.mpl_connect('motion_notify_event', crosshair.on_move)
	cID2 = fig1.canvas.mpl_connect('axes_enter_event', cursor.in_axes)
	# 'd' (done) and 'l' (toggle reference line); replaces Cursor.on_press whose 'l' is broken.
	cID3 = fig1.canvas.mpl_connect('key_press_event', on_key)
	cID10 = fig2.canvas.mpl_connect('key_press_event', on_key)
	# Arrow-key epoch navigation, active whether the main or zoomed figure has focus.
	cID5 = fig1.canvas.mpl_connect('key_press_event', on_arrow_key)
	cID6 = fig2.canvas.mpl_connect('key_press_event', on_arrow_key)
	# Click an epoch's state in the zoomed strip to relabel it from the terminal.
	cID7 = fig2.canvas.mpl_connect('button_press_event', on_state_strip_click)
	# Press 'v' (when no clip is open) to play just the current epoch's video.
	cID8 = fig1.canvas.mpl_connect('key_press_event', on_video_key)
	cID9 = fig2.canvas.mpl_connect('key_press_event', on_video_key)

	# Main event loop: stay open, redrawing traces and applying state edits, until DONE.
	plt.show()
	DONE = False
	while not DONE:
		# Process GUI clicks/keys for a short window, then fall through to poll the terminal.
		# The timeout keeps the loop ticking (so strip selections can be applied) without
		# stealing keyboard focus from the terminal the way a focus-raising redraw would.
		plt.waitforbuttonpress(timeout=0.2)

		# If epochs are picked on the strip and the user has typed a line in the terminal,
		# apply that state to all of them at once. stdin is only touched while a selection is
		# pending, so the blocking prompts elsewhere (e.g. range edits) keep working normally.
		if strip_selection:
			try:
				stdin_ready = bool(select.select([sys.stdin], [], [], 0)[0])
			except (OSError, ValueError):
				stdin_ready = False  # stdin isn't selectable (e.g. not a real terminal)
			if stdin_ready:
				line = sys.stdin.readline().strip()
				if line != '':
					try:
						new_state = int(line)
					except ValueError:
						print(f"'{line}' is not a valid state; selection kept.")
						new_state = None
					if new_state is not None:
						for idx in sorted(strip_selection):
							State[idx] = new_state
							# Recolor exactly this one epoch on the main hypnogram. Width 1 = one
							# epoch, so it stays aligned with the zoomed strip / State.
							# SWS_utils.correct_bins drew a 1.5-wide bar that bled into the next
							# epoch ("one epoch ahead"), so we draw the rectangle directly instead.
							ax2.add_patch(patch.Rectangle((idx, 0), 1, height=1,
								color=state_to_color(new_state)))
						np.save(os.path.join(d['savedir'],
							'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)
						print(f'Changed {len(strip_selection)} epoch(s) to state {new_state}.')
						strip_selection.clear()
						draw_state_strip(ax_state, State, this_epoch_t, start_trace, end_trace,
							d['epochlen'], selected=strip_selection)
						fig2.canvas.draw()
						fig1.canvas.draw()

		# The user clicked a new location: recenter the zoomed traces (and video) on that epoch.
		if cursor.replot:
			print("Replot of fig 1. called!")
			this_epoch_t = math.floor(cursor.replotx / d['epochlen']) * d['epochlen']
			replot_start = start_trace + this_epoch_t
			replot_end = end_trace + this_epoch_t
			print('Epoch Start Time = ' + str(this_epoch_t) + ' seconds')
			print('Start Trace = ' + str(replot_start) + ' seconds')
			print('End Trace = ' + str(replot_end) + ' seconds')

			SWS_utils.update_raw_trace(fig1, fig2, line1, line2, line3, line4, line5, long_emg,
				long_emg_t, long_ThD, long_ThD_t, long_v, long_v_t, markers, this_epoch_t,
				replot_start, replot_end, d['epochlen'])

			# Redraw the state strip for the epochs now centered in the zoomed window
			# (keeping any pending selection visible as it scrolls into/out of view).
			draw_state_strip(ax_state, State, this_epoch_t, start_trace, end_trace, d['epochlen'],
				selected=strip_selection)
			fig2.canvas.draw()

			# Pull up the corresponding window of behavior video if video review is on,
			# but skip it when the move came from arrow-key navigation.
			if d['vid'] and not arrow_nav[0]:
				if this_epoch_t - d['epochlen'] < 0:
					print('No video available for this bin')
				else:
					vid_start = int(this_timestamp.index[this_timestamp['Offset_Time'] > (this_epoch_t - d['epochlen'])][0])
					vid_end = int(this_timestamp.index[this_timestamp['Offset_Time'] < ((this_epoch_t) + (d['epochlen'] * 2))][-1])
					SWS_utils.pull_up_movie(d, cap, vid_start, vid_end,
						this_video, d['epochlen'], this_timestamp)

			arrow_nav[0] = False  # reset so the next spectrogram click can show video again
			plt.show()
			cursor.replot = False

		# The user selected a range of epochs and wants to relabel them.
		if cursor.change_bins:
			bins = np.sort(cursor.bins)
			start_bin = cursor.bins[0]
			end_bin = cursor.bins[1]
			print(f'changing bins: {start_bin} to {end_bin}')
			SWS_utils.clear_bins(bins, ax2)
			fig2.canvas.draw()

			# Prompt for the new state; re-prompt once if the input can't be parsed as an int.
			try:
				new_state = int(input('What state should these be?: '))
			except ValueError:
				new_state = int(input('What state should these be?: '))

			SWS_utils.correct_bins(start_bin, end_bin, ax2, new_state)
			fig2.canvas.draw()
			State[start_bin:end_bin] = new_state
			# Slicing is end-exclusive, so explicitly set the last bin if it's the final epoch.
			if end_bin == len(State) - 1:
				State[end_bin] = new_state
			np.save(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)

			# Reflect the just-edited states in the zoomed window's state strip.
			draw_state_strip(ax_state, State, this_epoch_t, start_trace, end_trace, d['epochlen'],
				selected=strip_selection)
			fig2.canvas.draw()

			cursor.bins = []
			cursor.change_bins = False

		# The user pressed the key that ends scoring for this segment.
		if cursor.DONE:
			DONE = True

	# Clean up windows and persist the final state array before returning.
	print('successfully left GUI')
	cv2.destroyAllWindows()
	plt.close('all')
	np.save(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'), State)

	return State


def start_swscoring(d):
	"""Top-level driver: pick an acquisition, score each hour of it, and optionally retrain/log.

	`d` is the settings dictionary. The function:
	  1. Lists acquisitions that already have a saved State file.
	  2. Asks the user which acquisition to score and loads its EEG/EMG.
	  3. Splits the acquisition into ~1-hour segments and, for each one, builds the
	     feature dictionary, then either corrects existing scoring ('c') or scores a
	     new dataset ('s', with or without the random-forest model).
	  4. After scoring, optionally retrains the model and writes log entries.
	"""
	# These older dependencies emit a lot of deprecation noise; silence it.
	print('this code is supressing warnings')
	warnings.filterwarnings("ignore")
	print('These are the available acquisitions: ' + str(d['Acquisition']))

	# Scan the save directory for already-scored acquisitions so the user knows what's done.
	state_files = glob.glob(os.path.join(d['savedir'], 'StatesAcq*.npy'))
	scored_acqs = []
	for sf in state_files:
		filename = os.path.split(sf)[1]
		idx1 = filename.find('q')   # the 'q' in "...Acq"
		idx2 = filename.find('_')   # the underscore before "hr"
		try:
			acq_num = int(filename[idx1 + 1:idx2])
		except ValueError:
			continue
		scored_acqs.append(acq_num)
	print('These are the acquisitions that have a previous State file: ' + str(sorted(scored_acqs)))
	a = input('Which acqusition do you want to score?')

	# Load the full downsampled EEG (and EMG) to figure out how long the acquisition is.
	print('Loading EEG and EMG....')
	downsampEEG = np.load(os.path.join(d['savedir'], 'downsampEEG_Acq' + str(a) + '.npy'))
	if d['emg']:
		downsampEMG = np.load(os.path.join(d['savedir'], 'downsampEMG_Acq' + str(a) + '.npy'))
	acq_len = np.size(downsampEEG) / d['fsd']  # acquisition length in seconds (fsd = downsampled rate)
	hour_segs = math.ceil(acq_len / 3600)      # number of ~1-hour segments to score
	print('This acquisition has ' + str(hour_segs) + ' segments.')

	# Wall-clock start time of the acquisition, used to align video to the EEG.
	acq_start = SWS_utils.get_AcqStart(d, a, acq_len)

	# Score one hour-long segment at a time.
	for h in np.arange(hour_segs):
		# Assemble this segment's EEG channels and EMG into one dataframe, and collect
		# the per-channel normalization values used during feature extraction.
		eeg_df = pd.DataFrame()
		normVal = []
		for e in d['EEG channel']:
			eeg_dir = os.path.join(d['savedir'], 'AD' + str(e) + '_downsampled')
			eeg_df['EEGChannel' + str(e)] = np.load(os.path.join(eeg_dir, 'downsampEEG_Acq' + a + '_hr' + str(0) + '.npy'))
			normVal.append(np.load(os.path.join(eeg_dir, d['basename'] + '_normVal.npy')))

		eeg_df['EMG'] = np.load(os.path.join(d['savedir'], 'downsampEMG_Acq' + str(a) + '_hr' + str(h) + '.npy'))

		# Trim the tail so the segment is an exact multiple of the epoch length.
		seg_len = len(eeg_df) / d['fsd']
		nearest_epoch = math.floor(seg_len / d['epochlen'])
		new_length = int(nearest_epoch * d['epochlen'] * d['fsd'])
		eeg_df = eeg_df.iloc[:new_length]

		# Build the per-epoch feature dictionary the classifier/GUI consume.
		FeatureDict = SWS_utils.build_feature_dict(eeg_df, d['fsd'], d['epochlen'], normVal=normVal)

		# Load/initialize the behavior video and movement (velocity) tracking for this segment.
		this_video, v, this_motion = SWS_utils.initialize_vid_and_move(d, a, acq_start, acq_len)
		if d['movement']:
			FeatureDict['Velocity'] = v[0]
		FeatureDict['animal_name'] = np.full(len(FeatureDict[list(FeatureDict.keys())[0]]), d['mouse_name'])

		os.chdir(d['savedir'])

		# The EMG trace for this segment; defined once here so it's available in every branch below.
		this_emg = eeg_df['EMG']

		# Either correct an existing scoring or score a new one.
		check = input('Do you want to check and fix existing scoring (c) or score new dataset (s)?: c/s ')
		while check != 'c' and check != 's':
			check = input(
				'Only c/s is accepted. Do you want to check and fix existing scoring (c) or score new dataset (s)?: c/s ')

		if check == 'c':
			try:
				# Load the previously-saved states, treat NaNs as unscored (0), and open the GUI.
				State = np.load(os.path.join(d['savedir'], 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'))
				wrong, = np.where(np.isnan(State))
				State[wrong] = 0
				State = display_and_fix_scoring(d, a, h, this_emg, State, False, None,
										None, this_video, acq_start, v=v, movement_df=this_motion)
				# Warn about any epochs still left unscored and offer to fix them immediately.
				if np.any(State == 0):
					print('The following bins are not scored: \n' + str(np.where(State == 0)[0]))
					zero_check = input('Do you want to go back and fix this right now? (y/n)') == 'y'
					if zero_check:
						State = display_and_fix_scoring(d, a, h, this_emg, State, False, None,
										None, this_video, acq_start, v=v, movement_df=this_motion)
					else:
						print('Ok, but please do not update the model until you fix them')
			except FileNotFoundError:
				# Nothing to correct yet for this segment.
				print("There is no existing scoring.")

		elif check == 's':
			# Optionally seed the GUI with random-forest predictions before manual review.
			model = input('Use a random forest? y/n: ') == 'y'

			if model:
				jobname = SWS_utils.build_joblib_name(d)
				try:
					clf = joblib.load(os.path.join(d['model_dir'], jobname))
				except FileNotFoundError:
					print("You don't have a model to work with.")
					return

				# Build the feature matrix, predict states, clean up the prediction, and save it.
				Features = SWS_utils.prepare_feature_data(FeatureDict, d['movement'])
				Predict_y = clf.predict(Features)
				Predict_y = SWS_utils.fix_states(Predict_y)
				np.save(os.path.join(d['savedir'], 'model_prediction_Acq' + str(a) + '_hr' + str(h) + '.npy'), Predict_y)
				State = display_and_fix_scoring(d, a, h, this_emg, Predict_y, True, clf,
					Features, this_video, acq_start, v=v, movement_df=this_motion)
			else:
				# Start from an all-zero (unscored) array and score entirely by hand.
				State = np.zeros(int(acq_len / d['epochlen']))
				State = display_and_fix_scoring(d, a, h, this_emg, State, False, None,
										None, this_video, acq_start, v=v, movement_df=this_motion)

		# Attach the final human-validated states to the feature dictionary for retraining.
		FeatureDict['State'] = State

		# Optionally fold this segment into the model and record what was done.
		update = input('Do you want to update the model?: y/n ') == 'y'
		if update:
			update_model(d, FeatureDict)
			model_log(d['modellog_dir'], 0, d['species'], d['mouse_name'], d['mod_name'], a)
		logq = input('Do you want to update your personal log?: y/n ') == 'y'
		if logq:
			personal_log(d['personallog_dir'], d['mouse_name'], d['savedir'], a)

		plt.close('all')


def load_data_for_sw(filename_sw, return_data=False):
	"""Load a Score_Settings.json file and either return the settings or start scoring.

	`filename_sw` is the path to the JSON settings file. If `return_data` is True the
	parsed settings dictionary is returned (useful for callers that just need config);
	otherwise scoring begins immediately via start_swscoring.
	"""
	with open(filename_sw, 'r') as f:
		d = json.load(f)
	if return_data:
		return d
	start_swscoring(d)


def model_log(log_dir, action, animal, mouse_name, mod_name, a):
	"""Append a human-readable line to the shared model scoring log.

	Records that a given acquisition was corrected / scored-with-model / scored-in-legacy
	(selected by `action`: 0/1/2) for a given animal, by whoever is at the keyboard, with a
	timestamp. The log file (`{mod_name}_scoringlog.txt` in `log_dir`) is created if missing.
	"""
	log_file = os.path.join(log_dir, mod_name + '_scoringlog.txt')
	if not os.path.exists(log_file):
		print(log_file + ' does not exist. Making it now')
		f = open(log_file, "w+")
		f.close()

	# Maps the numeric action code to the phrase written into the log.
	state_dict = {'0': 'corrected',
					'1': 'scored with ML model',
					'2': 'scored in legacy mode'}

	print("Logging to " + log_file)
	file = open(log_file, "a+")

	# Current date/time stamped onto the log entry (mm/dd/YYYY HH:MM:SS).
	now = datetime.now()
	dt_string = now.strftime("%m/%d/%Y %H:%M:%S")

	whois = input("What is your name?:")
	file.write(animal + " " + mouse_name + " acquisition " + str(a) + " was " +
		state_dict[str(action)] + " by " + whois + " on " + dt_string + "\n")
	file.flush()
	file.close()


def personal_log(log_dir, mouse_name, save_dir, a):
	"""Append a row to the user's personal scoring log CSV.

	Records the date, mouse name, acquisition, and where the State array was saved, in
	`personal_scoringlog.csv` inside `log_dir`. Creates the CSV with a header if it
	doesn't exist yet, then appends one row without re-writing the header.
	"""
	log_file = os.path.join(log_dir, 'personal_scoringlog.csv')
	if not os.path.exists(log_file):
		print(log_file + ' does not exist. Making it now')
		df = pd.DataFrame(columns=['Date', 'Mouse Name', 'Acquisition', 'State Array Location'])
		df.to_csv(log_file, mode='a', header=True, index=False)
	d = {'Date': [pd.Timestamp.now()], 'Mouse Name': [mouse_name], 'Acquisition': [a], 'State Array Location': [save_dir]}
	df = pd.DataFrame(data=d)
	df.to_csv(log_file, mode='a', header=False, index=False)


if __name__ == "__main__":
	# Command-line entry point: expects exactly one argument, the path to Score_Settings.json.
	args = sys.argv
	if len(args) < 2:
		print("You need to specify the path of your Score_Settings.json. For instance, run `python New_SWS.py /home/ChenLab_Sleep_Scoring/Score_Settings.json`.")
	elif len(args) > 2:
		print("You only need to specify the path of your Score_Settings.json. For instance, run `python New_SWS.py /home/ChenLab_Sleep_Scoring/Score_Settings.json`.")
	else:
		load_data_for_sw(args[1])
