# Code Review — `zbr_gui_v2`

Pinned to commit `3281000` (2026-10-26). Line numbers refer to that commit.
**Nothing was deleted or changed for this review** — every item is a proposal.

**Method.** An AST pass over every module (definitions with no references, unused
and duplicate imports, commented-out code), a search of every caller across
`~/Documents/git` (`PKA_Sleep`, `chenlab/` notebooks, …), and — for every claimed
bug — running the code path, not just reading it. Each finding says which.

---

## 0. Read this first: the package is a shared library

`SWS_utils`, `extract_data` and `extract_data_zbr` are imported by lab analysis
code outside this repo. Any cleanup must keep these names and import paths working:

| Name | Used by |
|---|---|
| `SWS_utils.peak_freq`, `SWS_utils.bandPower` | `PKA_Sleep/GLM.py`, `chenlab/rotation/*.ipynb` |
| `SWS_utils.define_microarousals` | `PKA_Sleep/Feedback_SS_Align.py` |
| `SWS_utils.timestamp_extracting` | `PKA_Sleep/AlarmClock_Analyses.py`, closed-loop notebooks |
| `SWS_utils.sort_timestamp_files`, `rename_DLC_csvs`, `DLC_check_fig` | `video_phys_timestamp_alignment.ipynb` |
| `extract_data.get_phys_fs`, `save_to_edf`, `make_edf_file` | `video_phys_timestamp_alignment.ipynb`, `save_as_edf.ipynb` |
| `extract_data_zbr.make_trimmed_edf_file` | `lsn_realtime_accuracy.ipynb` (your uncommitted EDF work) |

Note that notebooks import from **both** `extract_data` and `extract_data_zbr`.

---

## 1. Bugs (ranked)

| # | Severity | Where | Problem | Verified |
|---|---|---|---|---|
| 1 | **High** | [`New_SWS.py#L456`](neuroscience_sleep_scoring/New_SWS.py#L456) | **"Update model after scoring" always crashes.** `adjust_movement(FeatureDict, d['movement'], epochlen=…)` → `TypeError: got multiple values for argument 'epochlen'`. States are already saved, so no scoring is lost, but the model is never updated, and the exception skips the personal log and any later hour segments. | Ran |
| 2 | **High** (trap) | [`New_SWS.py#L1287`](neuroscience_sleep_scoring/New_SWS.py#L1287) | Fixing #1 exposes `model_log`'s `input("What is your name?")` — a terminal prompt in the GUI flow, so the GUI will appear frozen. Fix both together. | Read |
| 3 | **High** — *introduced by me* | [`__init__.py#L34`](neuroscience_sleep_scoring/__init__.py#L34), [`New_SWS.py#L6`](neuroscience_sleep_scoring/New_SWS.py#L6) | The macOS TkAgg pin runs at **import time**, so on a Mac any notebook doing `from neuroscience_sleep_scoring import SWS_utils` is forced off its inline backend. Should run only when the GUI launches. | Read |
| 4 | Medium | [`SWS_utils.py#L992`](neuroscience_sleep_scoring/SWS_utils.py#L992), L1012, L1020, L1451 | `sys.exit()` inside library functions (missing videos/timestamps, file-count mismatch) **kills the launcher and every open window** instead of reporting. Also hits `PKA_Sleep` callers. | Read |
| 5 | Medium | [`New_SWS.py#L932`](neuroscience_sleep_scoring/New_SWS.py#L932) | `except Exception` around `start_event_loop` catches **any** error raised by a callback, prints "start_event_loop unavailable" and silently switches to the old stuttering polling loop. Hit during this session's testing. | Ran |
| 6 | Medium | [`SW_Cursor.py#L249`](neuroscience_sleep_scoring/SW_Cursor.py#L249) | `c` moves only the dark-blue marker and sets `current_epoch_t` to the **unsnapped** mouse time, without a replot. The detail pane, overview markers and bin labels stay put. | Read |
| 7 | Latent | [`New_SWS.py#L1067`](neuroscience_sleep_scoring/New_SWS.py#L1067) | Features for hour `h` read EEG from **`_hr0`** while EMG reads `_hr{h}` (L1070). Every acquisition so far is exactly one hour, so it has never fired; a longer one would get predictions for hour 1+ from hour 0's EEG. | Read |
| 8 | Latent | [`SWS_utils.py#L1469`](neuroscience_sleep_scoring/SWS_utils.py#L1469) | `get_AcqStart` indexes `trigger_times.mat` by position in the settings' `Acquisition` list. If that list omits any acquisition (e.g. `choosing_acquisition`'s length filter), later acquisitions may get a neighbour's start time → wrong video/movement sync. Only recordings with `trigger_times.mat`; worth checking against one. | Read |
| 9 | Latent | [`SWS_utils.py#L1605`](neuroscience_sleep_scoring/SWS_utils.py#L1605) | `print_instructions` contains `'\.'` in a normal string: a `SyntaxWarning` today, a `SyntaxError` in a future Python — at which point **`SWS_utils` stops importing**, for the GUI and `PKA_Sleep` alike. Dead function; one-character fix (raw string) or remove. | Ran |
| 10 | Latent | [`SWS_utils.py#L694`](neuroscience_sleep_scoring/SWS_utils.py#L694) | `get_xfeatures(..., emg_flag=0)` reads `x_features` before assigning it → `NameError`. No caller passes `emg_flag`. | Read |
| 11 | Low | [`New_SWS.py#L1213`](neuroscience_sleep_scoring/New_SWS.py#L1213) | `build_model` cannot run: undefined `h` (L1230), `'AD' + str([0, 2])` path (L1222), EMG loaded from the EEG folder (L1232), a hard-coded `/Volumes/yaochen/Active/Lizzie/...` path (L1240). | Read |
| 12 | Low | `update_model` | Appends `_2chan` to `mod_name` even if the settings already say `mouse_2chan` (the example JSON does) → `mouse_2chan_2chan`. | Read |
| 13 | Low | [`New_SWS.py#L448`](neuroscience_sleep_scoring/New_SWS.py#L448) | Legacy module-level `on_press` uses `np.float` (removed in NumPy 1.24). Never connected, so dead rather than live. | Read |

---

## 2. Dead and unused code

### Files

| File | Lines | State | Proposal |
|---|---|---|---|
| `extract_data_kg.py` | 453 | **Does not parse** — five unresolved merge-conflict blocks (L295–446) from a squashed commit | Archive or delete |
| `New_SWS_kg.py` | 949 | Calls `SWS_utils.clear_bins`/`correct_bins` (L697), which no longer exist; fails at the first correction. Its useful part (the state strip) is already ported. | Archive (git history keeps it) |
| `train_model.py` | 187 | Unimportable: bare `import SWS_utils` | Fix the import or archive |
| `checking_timestamps.py`, `filewrite.py` | 77 | Ad-hoc scripts, no functions | Move to a `scripts/` folder or delete |
| `testfile`, `log.txt` | 0 | Empty | Delete |
| `runscore.sh` | 1 | Invokes the old terminal entry point | Update to the launcher or delete |
| `README.md`, `Sleep_Scoring_README.pdf` | — | Describe the pre-launcher terminal workflow | Point to `SCORING_GUIDE.md` |
| `Score_Settings.json` (in package) | — | An example with another user's `/Volumes/...` paths | Rename to `Score_Settings.example.json` |
| `sleep score correct all state.ipynb` | — | One-off notebook inside the package | Move out of the package |

### Functions and classes with no caller anywhere (package + `PKA_Sleep` + notebooks)

| Where | Item | Lines | Note |
|---|---|---|---|
| `SW_Cursor.py#L1151` | `class ScoringCursor` | 161 | Imported by `New_SWS`, never instantiated |
| `SWS_utils.py#L735` | `pull_up_movie` | 64 | Still uses OpenCV HighGUI — the macOS crash path; never wire back |
| `SWS_utils.py#L1605` | `print_instructions` | 64 | Outdated text **and** bug #9 |
| `SWS_utils.py#L861` | `create_scoring_figure` | 38 | Superseded by `create_prediction_figure` |
| `SWS_utils.py` | `model_feature_importance`, `transfer_DLC_files`, `get_movement_segs`, `get_eeg_segment`/`get_eeg_file` | ~70 | |
| `New_SWS.py` | `on_press` (module), `key_stroke`, `build_model`, `load_data_for_sw`, `start_swscoring` | ~90 | The legacy terminal path; `start_swscoring` is reachable only through `load_data_for_sw` |
| `New_SWS.py#L46` | `_state_popup_root` | — | Always `None` since the popup rewrite; still checked in `ScoringLauncher.on_close` |
| `extract_data*.py` | `alternate_label`, `full_EEG_EMG` | ~16 | In both copies |

**Keep despite no internal caller** (public API, §0): `peak_freq`, `define_microarousals`,
`timestamp_extracting`, `get_phys_fs`, `make_trimmed_edf_file`, `rename_DLC_csvs`,
`DLC_check_fig`, `sort_timestamp_files`, `save_to_edf`, `make_edf_file`.

### Inside `Cursor` (`SW_Cursor.py`)

- **Never called:** `on_move`, `set_cross_hair_visible`, `_toggle_preview_window`, `crosshair`, `pull_up_movie`.
- **Written, never read:** `lines` and `toggle_line` — which also **draw two invisible lines at x=0** on two overview panels ([L151](neuroscience_sleep_scoring/SW_Cursor.py#L151)); `movement_x_axis`, `spect_x_axis` (hard-coded `linspace(199, 1442, 900)`), `movie_bin`, `second_click`, `_help_window`. (`_video_photo` is a false positive: holding the reference keeps the Tk image alive.)
- **Dead branch:** `movie_mode` is only set to `True` in commented-out code, so the `if self.movie_mode:` branch in `on_click` ([L1088](neuroscience_sleep_scoring/SW_Cursor.py#L1088)) never runs.
- **`in_axes`** ([L1001](neuroscience_sleep_scoring/SW_Cursor.py#L1001)) is wired to every axes-enter event but only prints under `DEBUG`, above ~30 lines of commented-out `mplcursors` experiments.

### Imports and commented-out code

- **Unused imports** — `New_SWS`: `mpimg`, `signal`, `cv2`, `RandomForestClassifier`, `accuracy_score`, `dt`, `ScoringCursor`, `pathlib`, `time`, `timedelta`, `io`; `SWS_utils`: `mpimg`, `cp`, `preprocessing`, `parse`, `tzinfo`, `UTC`; `extract_data*`: `check` (from `tabnanny`), `psutil`, `time`.
- **Duplicate imports** — `datetime` in `New_SWS`; `json` in both `extract_data` files.
- **Commented-out code** — 72 lines in `SW_Cursor` (blocks at L1017–1031, L1205–1278), 27 in `SWS_utils` (L328–335, L1115–1141, L1479–1484).

---

## 3. Duplication

| What | Size | Proposal |
|---|---|---|
| `extract_data.py` vs `extract_data_zbr.py` | 28 differing lines of ~500 (≈95% identical) | One implementation in `extract_data`; `extract_data_zbr` does `from .extract_data import *` and adds `precompute_spectrograms` and the EDF trimming. **Both import paths keep working** (§0). |
| Fig 1 crosshair blit | 3 copies: `_redraw_crosshair` (L679), `on_mouse_move_fig2` (L631), `_on_video_slider` (L821) | The other two call `_redraw_crosshair()`. Today's animated-marker change had to be added in all three places — exactly the cost of the duplication. |
| Modal dialog boilerplate | ~40 lines shared by `choose_state_popup` and `_ask_yes_no` (focus save/restore, grab, `bind_all`, `wait_variable`) | One `_run_modal(win, var)` helper |
| Detail-pane placement | `update_raw_trace` sets fig 2 limits to ±600 s and moves `line4`/`line5` (`SWS_utils.py#L932`); `align_detail_xaxes` immediately overwrites both | Let `align_detail_xaxes` own fig 2 |
| Zoomed-trace slicing | written in `update_raw_trace` and again in `magnify_update` (`New_SWS.py#L669`) | One `_slice_window(t, y, t0, t1)` |
| "Load this hour's EEG/EMG" | `score_acquisition`, `precompute_acq_features`, `display_and_fix_scoring` | One loader, which also fixes bug #7 in one place |
| File paths | `'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'` built by hand 5× | `states_path(d, a, h)`, like the existing `_recovery_path` |

---

## 4. Readability and structure (ranked by payoff)

1. **Split `display_and_fix_scoring`** — 529 lines with 9 nested functions sharing state through closures and `nonlocal`. A `ScoringSession` class holding `State`, `this_epoch_t`, the figures and artists, with methods `replot()`, `change_bins()`, `set_bin_labels()`, `align_detail()`, `save()`. Each piece becomes readable and testable on its own. Highest payoff, highest risk — do it last, with tests (item 9).
2. **Make the package import light.** `__init__` star-imports `New_SWS` and `extract_data`, so `from neuroscience_sleep_scoring import SWS_utils` takes **~1.5 s** and loads tkinter, OpenCV, scikit-learn, the GUI and `PKA_Sleep` (measured). Import the GUI only from its entry points; this also fixes bug #3.
3. **Replace `from pylab import *`** in `SWS_utils` ([L19](neuroscience_sleep_scoring/SWS_utils.py#L19)) with explicit imports. It silently replaces `all`, `any`, `sum` and `divmod` with NumPy versions — the source of the generator bug fixed earlier in this branch.
4. **Exceptions instead of `sys.exit()`** in library code (bug #4); the launcher already catches errors per acquisition.
5. **Fewer silent `except: pass`** — 70 across the four GUI modules. Most guard cosmetics and are fine; the ones around the event loop, saving and video should log.
6. **Validate settings once at load.** `d['…']` keys are read throughout; a missing key (e.g. `rpi`) currently surfaces mid-extraction. A small dataclass or check in `ScoringLauncher.load_settings` would fail fast with a clear message.
7. **Name the magic numbers** — 900 epochs per hour (`adjust_movement`), ±600 s, the asymmetric −16/+20 s trace window, `buffer=4`.
8. **One indentation style.** `New_SWS` and `extract_data*` use tabs; `SW_Cursor` and most of `SWS_utils` use spaces (with 19 tab lines mixed in).
9. **Commit the regression tests.** This session's tests (save → reopen round trip, key handling, launcher ZT labels) live only in scratch; a `tests/` folder would make items 1–3 safe to do.

---

## 5. Performance notes (measured)

| Path | Now | Note |
|---|---|---|
| Arrow step / click replot | 0.29 s | Was ~1.3 s before this commit. The rest is the detail pane's full redraw (~0.4 s at full window size). |
| `import SWS_utils` via the package | 1.5 s | See §4.2 |
| Opening an acquisition | — | Re-reads `All_movement.pkl` (140–175 MB) every time; `score_acquisition` loads the whole-acquisition EEG just for its length and loads the EMG without using it (`np.load(..., mmap_mode='r')` would make both free). |

---

## 5a. Addendum — found while adding scorer mode and view settings

| Finding | Status |
|---|---|
| **Spectrograms are drawn up to 5 s out of place at acquisition edges.** Columns are 10 s windows centred on 5 … T−5 s but the image is stretched across 0 … T, so content near the start is drawn ~5 s early and near the end ~5 s late (correct mid-acquisition). | **Fixed in the GUI** via an opt-in `true_time_extent=True`. **Not changed for analysis:** `plot_spectrogram`/`my_specgram` are used by `PKA_Sleep/cloop.py` and ~30 notebooks, whose figures carry the same edge offset. Your call whether to switch them. |
| **Trimmed acquisitions thrash the spectrogram cache.** The cache is keyed on signal length and the display trims dead EEG tails, so acquisitions with a dead tail (e.g. 20260706 acq 6) recompute and overwrite their cache on every open. | Open — performance only |
| The View Settings slider and its box shared one `IntVar`, so Tk rejected the box ever being empty ("can't assign non-numeric value to scale variable"): clearing it to type a number fought the user. | Fixed |
| `ScoringLauncher.launch_acq` overwrote an error message with "Done with Acq N", so a failed launch reported success. | Fixed |
| `add_buffer(..., fs=200)` is hard-coded for the EMG instead of using `d['fsd']`. | Open — latent (fsd is 200 everywhere today) |
| `plot_zoomed_data` draws the yellow current-epoch highlight once at t = 0, so it is only visible while viewing epoch 0. | Open — cosmetic |

## 6. Suggested order

1. **Correctness, no visible behavior change:** #1 + #2 together, #3, #4, #5, #9.
2. **`c` key** (#6): make it replot, like clicking the spectrogram.
3. **Light package import** (§4.2) — protects every notebook that uses `SWS_utils`.
4. **Commit the tests** (§4.9).
5. **Remove dead code** (§2), re-running the caller search first.
6. **Deduplicate** (§3), starting with `extract_data` while keeping both import paths.
7. **`ScoringSession` refactor** (§4.1).
