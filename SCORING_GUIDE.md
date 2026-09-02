# Sleep Scoring Guide (`zbr_gui_v2`)

A complete walkthrough of the rodent EEG/EMG sleep-scoring pipeline: where the data
comes from, how it is processed and stored, and how to operate the scoring GUI.

Written to be read start-to-finish once, then used as reference. If you are brand
new, read Parts 0–2, then do Part 8 (setup), then Part 5 (the GUI).

---

## Part 0 — Orientation

**What this software does.** A mouse is implanted with two EEG electrodes and one
EMG electrode and recorded for several hours while a camera films it. This package
takes those raw recordings, cuts them into 4-second **epochs**, and helps you label
each epoch as **Wake**, **NREM**, or **REM**. A random forest model makes a first
pass; you correct it in a GUI. The output is one array of state labels per
recording hour, which downstream analysis uses.

**The pipeline in one line:**

```
raw .mat + video + DLC csv  →  extract_data_zbr.py  →  extracted_data/  →  ScoringLauncher  →  StatesAcq*.npy
```

### Vocabulary

| Term | Meaning |
|---|---|
| **Acquisition** (`a`) | One continuous recording block, usually ~1 hour. Numbered (2, 3, 4…). The unit you pick in the launcher. |
| **Hour segment** (`h`) | If an acquisition runs longer than 3600 s it is split into hour segments `hr0`, `hr1`, … Most acquisitions are just `hr0`. |
| **Epoch** / **bin** | The scoring unit: 4 seconds (`epochlen`). A 3600 s hour = 900 epochs. |
| **State** | The label for one epoch. `1` Wake, `2` NREM, `3` REM, `0` unscored. |
| **`fs` / `fsd`** | Raw sampling rate (e.g. 1000 Hz) and downsampled rate (200 Hz). Everything after extraction is at `fsd`. |
| **AD channels** | Raw amplifier channels. `AD0` = EEG 1, `AD2` = EEG 2, `AD3` = EMG. (`AD1` is not used by this pipeline.) |
| **`savedir`** | The `*_extracted_data/` folder. Everything the GUI reads and writes lives here. |
| **Velocity** | Per-epoch movement speed, derived from DeepLabCut body-part tracking on the video. |

---

## Part 1 — Where the data comes from

Three independent recording streams have to be lined up in time. Understanding this
is most of understanding the pipeline.

### 1.1 Physiology (the EEG/EMG)

The acquisition computer writes one MATLAB `.mat` file **per channel per
acquisition** into `rawdat_dir`:

```
zbrDFCMbP002_20260706/
  AD0_2.mat  AD0_3.mat  AD0_4.mat  ...   ← EEG channel 1, acquisitions 2,3,4...
  AD1_2.mat  ...                          ← unused by this pipeline
  AD2_2.mat  ...                          ← EEG channel 2
  AD3_2.mat  ...                          ← EMG
```

Each file holds one continuous voltage trace at the raw rate `fs`. The signal is
buried in a nested MATLAB struct; the extraction code digs it out with
`scipy.io.loadmat(f)['AD0_2'][0][0][0][0]`. Some recordings were saved with a
memory-saving convention where the variable is just `AD0` rather than `AD0_2` —
the loader tries the normal key first and falls back automatically.

**When did the acquisition start?** This matters enormously, because it is how EEG
time gets matched to video time. `SWS_utils.get_AcqStart` tries three sources in
order:

1. `trigger_times.mat` in `rawdat_dir` — best.
2. `continuous*data_<a>.mat` → its `triggerTime` field — also good.
3. **Fallback:** the file modification time of `AD0_<a>.mat`, minus the acquisition
   length. This assumes the acquisition computer's clock was correct and that the
   file was written the instant recording ended. It prints a loud warning. If the
   video looks out of sync with the EEG, suspect this.

### 1.2 Video

A camera (Bonsai workflow, or a Raspberry Pi rig) records `.mp4` (or `.avi`) files
into `video_dir`, and for every video writes a **timestamp CSV** into `csv_dir`
with one wall-clock timestamp per frame:

```
zbrDFCMbP002_20260706_csvs/
  zbrDFCMbP002_20260706_timestamp000.csv
  zbrDFCMbP002_20260706_timestamp001.csv
  ...
```

Video files and timestamp files are matched by their trailing number, and sorted by
`SWS_utils.sort_files`. **The counts must match.** If there are 12 timestamp files
and 11 motion files, extraction stops with an error rather than silently
misaligning your data.

The timestamp CSVs are what convert "epoch 214 of acquisition 6" into "frame 8,930
of `videorec_…_vid002.mp4`". `pulling_timestamp` slices the frames falling inside
one acquisition's window and adds an `Offset_Time` column — seconds since the
acquisition started — which is exactly the x-axis the GUI plots on.

### 1.3 Movement (DeepLabCut)

DeepLabCut is run on the videos and writes a motion CSV per video into `csv_dir`
(`*_motion*.csv`) containing X, Y, and a `likelihood` for each tracked body part.
The settings key `DLC Label` picks which body part to use (e.g. `"center"`).

Velocity is then computed per epoch as straight-line displacement:
`v = sqrt(dx² + dy²)` between the first and last tracked frame of each 4 s bin
(`SWS_utils.movement_processing`). It is a coarse but effective "is the animal
moving" signal — high velocity is essentially incompatible with sleep, which makes
it the single most useful sanity check in the GUI.

Frames with likelihood < 0.8 are reported as "bad frames" during extraction. A high
percentage means the DLC model is struggling with that video, and velocity from it
should be trusted less.

---

## Part 2 — The settings file

Everything is driven by one JSON file, conventionally
`Score_Settings_<you>_<experiment>.json`, kept next to the raw data. **Make your own
copy per experiment** — never share one across experiments.

A real, working example:

```json
{
  "basename": "zbrDFCMbP002_20260706",
  "rawdat_dir": "/home/chenlab/ris/zach/flip/zbrDFCMbP002_20260706",
  "model_dir": "/home/chenlab/ris/CENTRAL_CODE/neuroscience_sleep_scoring_model/model/",
  "video_dir": ".../zbrDFCMbP002_20260706_videos/",
  "csv_dir": ".../zbrDFCMbP002_20260706_csvs/",
  "modellog_dir": "/home/chenlab/ris/CENTRAL_CODE/neuroscience_sleep_scoring_model/model/",
  "personallog_dir": "/home/chenlab/ris/zach/flip/",
  "species": "mouse",
  "mod_name": "mouse",
  "mouse_name": "5486-t4",
  "epochlen": 4,
  "fs": 1000,
  "fsd": 200,
  "emg": 1,
  "vid": 1,
  "movement": 1,
  "EEG channel": [0, 2],
  "EMG channel": 3,
  "Acquisition": [2, 3, 4, 5, 6],
  "Filter High": 100,
  "Filter Low": 0.5,
  "savedir": ".../zbrDFCMbP002_20260706_extracted_data",
  "Bonsai Version": 6,
  "DLC": 1,
  "DLC Label": "center",
  "Minimum_Frequency": 1,
  "Maximum_Frequency": 20,
  "vmin": "None",
  "vmax": "None",
  "rpi": 1
}
```

### Key reference

| Key | Type | What it controls |
|---|---|---|
| `basename` | str | Experiment name. Used to match video/CSV filenames and to name `normVal` files. Must match the actual filenames. |
| `rawdat_dir` | path | Where the raw `AD*.mat` files are. |
| `savedir` | path | Where extracted data + your scoring goes. Created if missing. |
| `video_dir`, `csv_dir` | path | Videos, and timestamp/motion CSVs. |
| `model_dir` | path | Holds the random forest `.joblib` and the training dataframe `*_model.pkl`. Shared lab-wide — treat as precious. |
| `modellog_dir`, `personallog_dir` | path | Text/CSV logs of who scored what. |
| `species`, `mod_name` | str | Naming for the model log and training set. Leave as `mouse`. |
| `mouse_name` | str | The animal, e.g. `5486-t4`. Stored with every training row. |
| `epochlen` | int | Scoring bin size in seconds. **4.** Do not go lower. |
| `fs` / `fsd` | int | Raw / downsampled rate. `fsd` = 200 throughout. |
| `emg`, `vid`, `movement` | 0/1 | Whether EMG / video / DLC movement exist. Turning one off removes that panel and drops those model features. |
| `EEG channel` | list | Which AD channels are EEG. `[0, 2]` for two-channel recordings — this also selects the two-channel model. |
| `EMG channel` | int | Usually `3`. |
| `Acquisition` | list[int] | Which acquisitions to process/score. Written for you by `choosing_acquisition`. |
| `Filter Low` / `Filter High` | float | EEG bandpass, 0.5–100 Hz. (EMG uses a fixed 10 Hz high-pass.) |
| `Minimum_Frequency` / `Maximum_Frequency` | float | Frequency range drawn in the spectrograms (e.g. 1–20 Hz). Display only. |
| `vmin` / `vmax` | num or `"None"` | Fixed spectrogram color limits. `"None"` (the string) = auto-scale per acquisition — recommended. |
| `DLC`, `DLC Label` | 0/1, str | Whether motion CSVs are DeepLabCut format, and which body part column to use. |
| `rpi` | 0/1 | Whether timestamps came from a Raspberry Pi rig (different CSV parsing). **Required** by `combine_bonsai_data` — extraction crashes without it. |
| `Bonsai Version` | int | Recorded for provenance; not read by current code. |

---

## Part 3 — Extraction: raw data → `extracted_data/`

Run **once per experiment**, before any scoring:

```bash
python -m neuroscience_sleep_scoring.extract_data_zbr /path/to/Score_Settings.json
```

It walks you through a series of y/n prompts. Here is what each step actually does.

### Step 1 — Choose acquisitions (`choosing_acquisition`)

Scans `rawdat_dir` for `AD0_*.mat` and asks which to include. Answer `y` (all),
`n` (pick one by one), `length` (only acquisitions above a minimum duration — handy
for dropping aborted 30-second recordings), or `range`. The result is **written back
into your JSON** as `Acquisition`.

### Step 2 — Downsample and filter (`downsample_filter`)

For each acquisition and each EEG channel:

- Band-pass the EEG: 3rd-order Butterworth, `Filter Low`–`Filter High` Hz, zero-phase
  (`filtfilt`).
- High-pass the EMG at 10 Hz.
- Resample from `fs` to `fsd` (1000 → 200 Hz).
- Save the whole acquisition, then split into hour segments and **truncate each
  segment to a whole number of 4 s epochs**.

Writes:

```
savedir/
  downsampEEG_Acq6.npy                        ← whole acquisition, channel from loop
  downsampEMG_Acq6.npy
  downsampEMG_Acq6_hr0.npy                    ← per hour segment
  AD0_downsampled/downsampEEG_Acq6_hr0.npy    ← per channel, per hour  ← what the GUI reads
  AD2_downsampled/downsampEEG_Acq6_hr0.npy
```

### Step 3 — Normalization value (`get_normalizing_value`)

Computes total spectral power across all `hr0` files and saves the **median** as
`AD0_downsampled/<basename>_normVal.npy` (and the same for AD2). Every band-power
feature is divided by this, so features are comparable across animals and
recording days despite differences in electrode impedance. If this file is missing,
scoring crashes on load.

### Step 4 — Precompute spectrograms (`precompute_spectrograms`) *(new in this branch)*

Optional but strongly recommended. Computes and caches the spectrograms, the
theta/delta ratio, and the per-epoch feature table for every acquisition, so the GUI
opens in a few seconds instead of ~30. See Part 4. This can also be done later from
the launcher.

### Step 5 — EDF export (`make_edf_file`) — optional

Concatenates AD0/AD2/AD3 into standard `.edf` files (default 250 Hz, 24 h chunks)
for viewing in external EEG software. Not used by the scoring GUI.

### Step 6 — Combine video/movement data (`combine_bonsai_data`)

Reads every timestamp CSV and motion CSV, aligns them, and writes two pickles:

```
savedir/All_timestamps.pkl     ← every video frame's wall-clock time + source file
savedir/All_movement.pkl       ← X, Y, likelihood, timestamp for every frame
```

These are what let the GUI jump from an epoch to the right frame of the right video
file. This step exits with an error if the timestamp and motion file counts differ.

### Step 7 — Full velocity array (`make_full_velocity_array`) — optional

Bins movement into 4 s velocity and saves `savedir/velocity_vector.npy` for the
whole experiment. Convenience for downstream analysis; the GUI computes velocity
per acquisition on its own.

### Resulting `savedir` layout

```
zbrDFCMbP002_20260706_extracted_data/
  AD0_downsampled/     downsampEEG_Acq*_hr*.npy, <basename>_normVal.npy
  AD2_downsampled/     downsampEEG_Acq*_hr*.npy, <basename>_normVal.npy
  downsampEEG_Acq*.npy         downsampEMG_Acq*[_hr*].npy
  All_timestamps.pkl           All_movement.pkl
  spectrogram_cache/           ← derived, safe to delete (Part 4)
  model_prediction_Acq*_hr*.npy  ← the model's first pass
  StatesAcq*_hr*.npy           ← YOUR SCORING. The real output.
  recovery/                    ← crash autosaves (Part 5.7)
```

---

## Part 4 — The cache layer

Everything in `savedir/spectrogram_cache/` is **derived** — deleting it costs time,
never data.

| File | Contents | Validated against |
|---|---|---|
| `spect_Acq6_hr0_AD0.npz` | Spectrogram `Pxx`, `freqs`, `bins` | signal length + spectrogram params |
| `thd_Acq6_hr0_AD2.npz` | Theta/delta ratio per second | signal length + `fsd` |
| `features_Acq6_hr0.joblib` | The full per-epoch feature table | row count of the EEG dataframe |

Each cache stores the parameters it was built with and recomputes automatically if
they don't match, so a stale cache cannot silently give you wrong data.

**Why it matters:** feature extraction alone took ~15 s per acquisition and video
file selection another ~12 s. Warm, opening an acquisition is a few seconds.

**Two ways to warm it:** step 4 of extraction, or the launcher's
**Precompute spectrograms…** button, which lets you select acquisitions and shows
progress. Precompute a whole experiment while you're at lunch.

---

## Part 5 — Using the GUI

### 5.1 Launching

```bash
python -m neuroscience_sleep_scoring.New_SWS /path/to/Score_Settings.json
```

The JSON path is optional — without it, the launcher opens with a **Browse** button.

The **launcher** window shows every acquisition as a button:

- 🟩 **green** — already has a `StatesAcq*.npy` file (scored)
- 🟦 **blue** — spectrogram cache ready (will open fast)
- ⬜ **grey** — neither

Set the options first, then click an acquisition to open it:

| Option | Effect |
|---|---|
| **Use random forest model** | Pre-label every epoch with the model, then you correct it. Applies to *Score new dataset* only. |
| **Score new dataset** / **Check/fix existing** | Start from the model's prediction, or load your existing `StatesAcq` file to review and fix. |
| **Update model after scoring** | Append this hour to the shared training set and retrain. **Off by default — leave it off** until you're confident (see 5.8). |
| **Update personal log after scoring** | Append a row to your `personal_scoringlog.csv`. On by default. |

**Close** exits the whole program (it also tears down stray figure windows). Window
positions are remembered in `~/.sleep_scoring_gui_layout.json` and restored next time.

### 5.2 The two windows

Opening an acquisition gives you an **overview** figure and a **detail** figure
(plus a video window if `vid: 1`).

**Overview (Figure 1) — the whole hour, five stacked panels sharing one x-axis:**

1. **EEG 1 spectrogram** (AD0) — frequency vs time, power as color.
2. **Predicted states** — one colored column per epoch, plus a black line showing
   model confidence. *This is the panel you click to score.* Its x-axis is in
   **epoch number**, not seconds.
3. **EEG 2 spectrogram** (AD2).
4. **Velocity** — movement per epoch.
5. **EMG amplitude** — muscle tone.

**Detail (Figure 2) — a zoomed window around the current epoch, five panels:**

1. **EEG 1 spectrogram**, ±span seconds around the current epoch (adjustable).
2. **EEG 2 spectrogram**, same span.
3. **Velocity**, zoomed to the trace window.
4. **EMG**, same window as velocity so the two line up exactly.
5. **Sleep state strip** — one colored box per visible epoch, current epoch outlined
   in yellow. **Click any box to relabel just that epoch.**

Every detail panel is centered on the **center of the current epoch** and labeled
*relative* to it: `0` is the middle of the epoch you're on, negative is earlier.

### 5.3 Reading the plots

| Signature | Usually means |
|---|---|
| High EMG + high velocity + broadband/low-power spectrogram | **Wake** |
| Low EMG, strong low-frequency (delta, 0.5–4 Hz) power | **NREM** |
| Low/flat EMG, strong theta (5–8 Hz), *no* movement | **REM** |
| Brief EMG/velocity spike inside a long sleep bout | **Microarousal** — a single Wake epoch (see the `m` key) |

REM almost always follows NREM, essentially never follows Wake directly. A lone REM
epoch surrounded by Wake is nearly always an error worth a second look.

### 5.4 Scoring: the core loop

1. **Click the overview spectrogram** to move the current epoch there. The detail
   window and the yellow marker follow. *(This no longer plays video — that's `o`.)*
2. **Inspect** the detail window: spectrograms, EMG, velocity, and if needed press
   `o` to watch the animal.
3. **Fix a run of epochs:** click the **start** epoch in the predicted-states panel,
   then click the **end** epoch. A popup asks Wake / NREM / REM (or press `1`/`2`/`3`).
4. **Fix a single epoch:** click its box in the detail window's state strip.
5. Repeat. When finished, press **`d`**.

Selections are forgiving: click order doesn't matter, `←`/`→` nudge the start after
the first click, and `r` or `Esc` cancels.

### 5.5 Keyboard reference

Keys work when either figure window has focus. All lowercase — no Shift.

| Key | Action |
|---|---|
| `click` ×2 | Select start and end epoch in the states panel, then choose the state |
| `1` `2` `3` | Answer the state popup: Wake / NREM / REM |
| `m` | Arm a **microarousal**: press again to grow the block by one epoch (unbounded), then click the states panel to drop that many **Wake** epochs |
| `o` | Play the **current epoch's** video once |
| `p` | Play **previous + current + next** epoch's video once |
| `g` | Toggle magnify mode (detail view follows the cursor live) |
| `v` | **View settings** — detail spectrogram x-span, magnify window, EMG y-limits. Each has a slider *and* a box for typing an exact number; click **Apply**. |
| `l` | Drop / remove a dashed reference line at the cursor across all overview panels |
| `c` | Move the current-epoch marker to the cursor |
| `←` `→` | Nudge the selection start (after the first click) |
| `r` / `Esc` | Cancel a pending selection or an armed microarousal |
| `i` | Show the shortcut list |
| `d` | **Done** — leave scoring and get the save prompt |

The video window shows a `t= / bin= / frame=` banner over each frame and opens at
the video's native size. Press `q` or `Esc` in it to cut playback short.

### 5.6 A note on trimmed displays

Acquisitions sometimes stop early, leaving a dead (flat or zero) tail in the EEG.
That tail wrecks spectrogram color scaling and desynchronizes the panels, so the GUI
detects where real data ends and trims the **display** to it — you'll see a message
like `Acq 6 hr 0: EEG goes dead at 2890s (of 3600s); trimming the display to real data.`

**This is display-only.** The saved State array keeps its full nominal length (900
epochs for an hour) regardless. Epochs past the end of real data stay `0`/unscored.

### 5.7 Saving and crash recovery

- Every correction is **autosaved** to `savedir/recovery/StatesAcq<a>_hr<h>.npy`.
- Pressing `d` asks **"Save sleep states for Acq a hr h?"**
  - **Yes** → writes the real file, `savedir/StatesAcq<a>_hr<h>.npy`.
  - **No** → your previous saved scoring is left untouched.
  - Either way the recovery file is cleared, because this is a clean exit.
- If the GUI crashed last time, reopening that acquisition offers to **recover** the
  unsaved work. Say yes unless you know it was garbage.

The recovery folder is deliberately a subdirectory so it never gets mistaken for real
scoring by the "which acquisitions are scored" scan.

### 5.8 Updating the model and logs

After you exit, depending on the launcher checkboxes:

- **Update model** appends this hour's features + your labels to the shared training
  dataframe in `model_dir` and **retrains the random forest for the whole lab.** It
  also asks your name for the model log. Do not do this with scoring you aren't sure
  about, and never with an hour that still contains unscored (`0`) epochs.
- **Update personal log** appends date / mouse / acquisition / location to
  `personallog_dir/personal_scoringlog.csv`. Harmless and useful.

---

## Part 6 — Features and the random forest

For each 4 s epoch and each EEG channel, `build_feature_dict` computes:

- **Band power**, normalized by `normVal`: Delta (0.5–4), Theta (5–8), Alpha (8–12),
  BroadTheta (2–16), Fire (4–20) Hz.
- **Ratios:** `thet_delt` (theta/delta) and `nb` (theta/broad-theta).
- **Signal variance** per epoch, for EEG and for EMG (`EMGvar`).
- **Temporal context:** each of delta, theta, and `nb` shifted forward and backward
  up to 3 epochs (`delta_pre`, `delta_post2`, …). This is why the model can use
  "what came before" — important, since REM is defined partly by context.
- **Velocity**, if `movement: 1`.

The model file is named for the configuration it was trained on, e.g.
`EEG_2chan_EMG_movement.joblib` — two EEG channels, with EMG, with movement. Your
settings must match a model that exists in `model_dir`, or you get
`"You don't have a model to work with."`

Its raw predictions are lightly smoothed by `fix_states` (removing implausible
one-epoch flickers) and saved as `model_prediction_Acq*_hr*.npy` before you touch
them, so you can always compare model vs. human.

Retraining trains on the first half, reports train/test accuracy to the terminal,
then refits on everything and overwrites the `.joblib`. Watch those accuracy numbers
if you do retrain.

---

## Part 7 — Outputs

The deliverable is one file per acquisition-hour:

```
savedir/StatesAcq6_hr0.npy
```

A 1-D NumPy array, one entry per epoch (900 for a full hour), in acquisition order:

| Value | State | Color in GUI |
|---|---|---|
| `0` | Unscored | grey |
| `1` | Wake | green |
| `2` | NREM | blue |
| `3` | REM | red |

To use it:

```python
import numpy as np
State = np.load('.../StatesAcq6_hr0.npy')
epochlen = 4
time_s = np.arange(len(State)) * epochlen      # start time of each epoch
nrem_fraction = (State == 2).mean()
```

**Check for `0`s before analyzing** — they mean unscored, not "quiet". Check-mode
tells you which epochs are still zero when you exit.

---

## Part 8 — Setting up on a new machine

### 8.1 Get both repositories

This package imports `PKA_Sleep`, a separate lab repo. It is a **hard requirement** —
without it, even the GUI fails to import.

```bash
git clone git@github.com:YaoChenLabWashU/neuroscience_sleep_scoring.git
```
```bash
git clone git@github.com:YaoChenLabWashU/PKA_Sleep.git
```
```bash
cd neuroscience_sleep_scoring && git checkout zbr_gui_v2
```

### 8.2 Environment

Python ≥ 3.8 with: `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-learn`,
`joblib`, `opencv-python`, `natsort`, `psutil`, `pyedflib`, `seaborn`, plus **Tk**
(`python3-tk` on Debian/Ubuntu — matplotlib's TkAgg backend and every dialog need it).

```bash
conda create -n sleepscoring python=3.10
```
```bash
conda activate sleepscoring && pip install numpy scipy matplotlib pandas scikit-learn joblib opencv-python natsort psutil pyedflib seaborn
```

Then install both packages in editable mode:

```bash
pip install -e /path/to/neuroscience_sleep_scoring
```
```bash
pip install -e /path/to/PKA_Sleep
```

(If `PKA_Sleep` has no `setup.py`, add its parent directory to `PYTHONPATH` instead.)

### 8.3 Point it at data

The scoring code needs a real display (it opens Tk and OpenCV windows) — it will not
run over a plain SSH session without X forwarding.

Copy a settings JSON and **edit every path in it** for the new machine, then verify:

```bash
python -c "import json,os; d=json.load(open('Score_Settings.json')); [print(('OK  ' if os.path.exists(d[k]) else 'MISSING '), k, d[k]) for k in ['rawdat_dir','savedir','video_dir','csv_dir','model_dir']]"
```

If `savedir` already contains extracted data, you can score immediately. Otherwise
run extraction (Part 3) first.

---

## Part 9 — Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `You don't have a model to work with.` | No `.joblib` in `model_dir` matching your EEG/EMG/movement configuration. Check `model_dir` and that `EEG channel`, `emg`, `movement` are right. |
| `FileNotFoundError: ..._normVal.npy` | Extraction step 3 was skipped. Run `get_normalizing_value`. |
| Extraction exits: "different number of timestamp files and movement files" | DLC hasn't been run on every video, or a file is missing. Fix the counts; don't work around it — it would misalign video with EEG. |
| `KeyError: 'rpi'` during extraction | Add `"rpi": 0` or `1` to your settings JSON. |
| Video won't play / "No video available" | `vid` is 0, or timestamps for this acquisition are missing (the GUI prints "turning off video access" and continues). Check `All_timestamps.pkl` exists and `video_dir`/`csv_dir` are correct. |
| Video plays but the animal doesn't match the EEG | Acquisition start time came from the file-modification-time fallback (Part 1.1). Check for `trigger_times.mat`. |
| Spectrogram looks washed out / all one color | Usually a dead tail; the current code trims it automatically. If it persists, set `vmin`/`vmax` explicitly in the JSON. |
| Acquisition opens slowly | Cold cache. Use **Precompute spectrograms…** in the launcher. |
| GUI freezes and won't close | Historically caused by multiple Tk roots. All dialogs now share matplotlib's root — if you're editing the code, **never create a second `tk.Tk()`**. |
| Scored acquisitions aren't showing green | The scan looks for `StatesAcq*.npy` directly in `savedir`. Files in `recovery/` deliberately don't count. Click **Refresh**. |

### For anyone editing the code

`SWS_utils.py` does `from pylab import *`, which **shadows the built-in `all()` and
`any()` with the NumPy versions**. `np.all(<generator>)` is always `True`. Always
pass a list.

The invariant to preserve: `StatesAcq{a}_hr{h}.npy` must stay byte-identical in
format. Display changes (trimming, colors, layout) are display-only; the State array
is the source of truth and keeps its full nominal length.

---

## Appendix — Quick reference card

```
LAUNCH     python -m neuroscience_sleep_scoring.New_SWS Score_Settings.json
EXTRACT    python -m neuroscience_sleep_scoring.extract_data_zbr Score_Settings.json

STATES     0 unscored (grey) · 1 Wake (green) · 2 NREM (blue) · 3 REM (red)
EPOCH      4 s · 900 epochs per hour

SCORE      click spectrogram → move here
           click start epoch, click end epoch → pick state
           click a box in the detail state strip → relabel one epoch
KEYS       1/2/3 state · m microarousal · o play bin · p play 3 bins
           g magnify · v view settings · l reference line · c marker
           r/Esc cancel · i help · d done

OUTPUT     savedir/StatesAcq<a>_hr<h>.npy
AUTOSAVE   savedir/recovery/StatesAcq<a>_hr<h>.npy
CACHE      savedir/spectrogram_cache/   (derived — safe to delete)
LAYOUT     ~/.sleep_scoring_gui_layout.json
```
