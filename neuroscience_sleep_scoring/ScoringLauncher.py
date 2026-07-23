"""Central tkinter launcher for the sleep-scoring GUI.

Replaces the chain of terminal input() prompts: pick a Score_Settings JSON, see
every acquisition as a button (green = already scored, blue = spectrogram cache
ready), set the usual choices as checkboxes, and click an acquisition to score
it. The scoring figures reopen in their saved positions (see SWS_utils layout
helpers) with the video already at frame 0.

Also lets you precompute spectrograms for all / selected acquisitions so the GUI
opens instantly, and has a Close button to exit safely.

stdout still carries all the scoring prints for debugging.
"""
import os
import json
import tkinter as tk
from tkinter import filedialog, font as tkfont

from neuroscience_sleep_scoring import SWS_utils

GRID_COLS = 10  # acquisitions per row in the button grid

# Button colors
COL_SCORED = '#77dd77'   # green: has a State file
COL_CACHED = '#bcd6f7'   # blue: spectrogram cache present (fast open)
COL_PLAIN = '#e6e6e6'    # neither


def _scored_acquisitions(d):
    try:
        from neuroscience_sleep_scoring.New_SWS import scored_acquisitions
        return scored_acquisitions(d)
    except Exception as e:
        print(f'Could not scan scored acquisitions: {e}')
        return []


class ScoringLauncher:
    def __init__(self, root, settings_path=None):
        self.root = root
        self.d = None
        self.settings_path = None
        self._acq_order = []
        root.title('Sleep Scoring Launcher')

        geo = SWS_utils.load_layout().get('launcher')
        if geo:
            try:
                root.wm_geometry(geo)
            except Exception:
                pass

        self.bold = tkfont.Font(weight='bold')

        # --- Settings file row ---
        top = tk.Frame(root)
        top.pack(fill='x', padx=10, pady=(10, 4))
        tk.Label(top, text='Settings JSON:', font=self.bold).pack(side='left')
        self.path_var = tk.StringVar(value='(none selected)')
        tk.Label(top, textvariable=self.path_var, anchor='w').pack(
            side='left', fill='x', expand=True, padx=6)
        tk.Button(top, text='Browse...', command=self.browse).pack(side='right')

        # --- Options ---
        opt = tk.LabelFrame(root, text='Options (applied when you click an acquisition)',
            padx=8, pady=6)
        opt.pack(fill='x', padx=10, pady=4)
        self.use_model = tk.BooleanVar(value=True)
        tk.Checkbutton(opt, text='Use random forest model',
            variable=self.use_model).grid(row=0, column=0, sticky='w', columnspan=2)
        self.mode = tk.StringVar(value='s')
        tk.Radiobutton(opt, text='Score new dataset', variable=self.mode,
            value='s').grid(row=1, column=0, sticky='w')
        tk.Radiobutton(opt, text='Check/fix existing', variable=self.mode,
            value='c').grid(row=1, column=1, sticky='w')
        self.update_model = tk.BooleanVar(value=False)
        tk.Checkbutton(opt, text='Update model after scoring',
            variable=self.update_model).grid(row=2, column=0, sticky='w', columnspan=2)
        self.update_log = tk.BooleanVar(value=False)
        tk.Checkbutton(opt, text='Update personal log after scoring',
            variable=self.update_log).grid(row=3, column=0, sticky='w', columnspan=2)

        # --- Acquisition button grid ---
        mid = tk.LabelFrame(root, text='Acquisitions  —  click one to launch', padx=8, pady=6)
        mid.pack(fill='both', expand=True, padx=10, pady=4)
        legend = tk.Frame(mid)
        legend.pack(fill='x', pady=(0, 6))
        self._legend_chip(legend, COL_SCORED, 'scored')
        self._legend_chip(legend, COL_CACHED, 'cache ready')
        self._legend_chip(legend, COL_PLAIN, 'not cached')
        self.grid_frame = tk.Frame(mid)
        self.grid_frame.pack(fill='both', expand=True)

        # --- Bottom bar ---
        bot = tk.Frame(root)
        bot.pack(fill='x', padx=10, pady=(4, 10))
        tk.Button(bot, text='Precompute spectrograms...', command=self.open_precache_dialog
            ).pack(side='left')
        tk.Button(bot, text='Refresh', command=self.refresh_acqs).pack(side='left', padx=6)
        tk.Button(bot, text='Close', font=self.bold, command=self.on_close).pack(side='right')
        self.status = tk.StringVar(value='Select a settings JSON to begin.')
        tk.Label(bot, textvariable=self.status, anchor='w').pack(side='left', padx=10)

        root.protocol('WM_DELETE_WINDOW', self.on_close)
        if settings_path:
            self.load_settings(settings_path)

    def _legend_chip(self, parent, color, text):
        f = tk.Frame(parent)
        f.pack(side='left', padx=(0, 12))
        tk.Label(f, width=2, bg=color, relief='groove').pack(side='left')
        tk.Label(f, text=text).pack(side='left', padx=3)

    # ----- settings / acquisition grid -----
    def browse(self):
        path = filedialog.askopenfilename(title='Select Score Settings JSON',
            filetypes=[('JSON', '*.json'), ('All files', '*.*')])
        if path:
            self.load_settings(path)

    def load_settings(self, path):
        try:
            with open(path, 'r') as f:
                self.d = json.load(f)
        except Exception as e:
            self.status.set(f'Failed to load: {e}')
            return
        self.settings_path = path
        self.path_var.set(path)
        self.refresh_acqs()

    def _acq_color(self, a, scored, cached):
        if a in scored:
            return COL_SCORED
        if a in cached:
            return COL_CACHED
        return COL_PLAIN

    def refresh_acqs(self):
        if self.d is None:
            return
        scored = set(_scored_acquisitions(self.d))
        cached = set(a for a in self.d.get('Acquisition', [])
                     if SWS_utils.spectrogram_cached(self.d, a))
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self._acq_order = list(self.d.get('Acquisition', []))
        for i, a in enumerate(self._acq_order):
            b = tk.Button(self.grid_frame, text=str(a), width=5,
                bg=self._acq_color(a, scored, cached), activebackground='#ffe08a',
                command=lambda a=a: self.launch_acq(a))
            b.grid(row=i // GRID_COLS, column=i % GRID_COLS, padx=2, pady=2, sticky='nsew')
        n_scored = len(scored & set(self._acq_order))
        n_cached = len(cached & set(self._acq_order))
        self.status.set(f'{len(self._acq_order)} acquisitions · {n_scored} scored · {n_cached} cached.')

    # ----- launch -----
    def launch_acq(self, a):
        if self.d is None:
            return
        self.status.set(f'Scoring Acq {a}... (see terminal)')
        self.root.update_idletasks()
        self._save_geometry()  # save now in case scoring crashes
        self.root.withdraw()
        try:
            from neuroscience_sleep_scoring import New_SWS
            New_SWS.score_acquisition(self.d, a,
                use_model=self.use_model.get(), mode=self.mode.get(),
                update_model_after=self.update_model.get(),
                update_log_after=self.update_log.get())
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status.set(f'Error scoring Acq {a}: {e}')
        finally:
            self.root.deiconify()
            self.refresh_acqs()
            self.status.set(f'Done with Acq {a}.')

    # ----- precompute dialog -----
    def open_precache_dialog(self):
        if self.d is None:
            self.status.set('Load a settings JSON first.')
            return
        acqs = list(self.d.get('Acquisition', []))
        cached = set(a for a in acqs if SWS_utils.spectrogram_cached(self.d, a))

        win = tk.Toplevel(self.root)
        win.title('Precompute spectrograms')
        win.transient(self.root)
        tk.Label(win, text='Select acquisitions to precompute (uncached are pre-checked):',
            font=self.bold).pack(padx=10, pady=(10, 4))

        grid = tk.Frame(win)
        grid.pack(padx=10, pady=4)
        vars_by_acq = {}
        for i, a in enumerate(acqs):
            v = tk.BooleanVar(value=(a not in cached))
            vars_by_acq[a] = v
            cb = tk.Checkbutton(grid, text=str(a), variable=v, width=4,
                bg=(COL_CACHED if a in cached else COL_PLAIN))
            cb.grid(row=i // GRID_COLS, column=i % GRID_COLS, padx=1, pady=1, sticky='w')

        def set_all(val):
            for v in vars_by_acq.values():
                v.set(val)

        def set_uncached():
            for a, v in vars_by_acq.items():
                v.set(a not in cached)

        sel = tk.Frame(win)
        sel.pack(fill='x', padx=10, pady=4)
        tk.Button(sel, text='All', command=lambda: set_all(True)).pack(side='left')
        tk.Button(sel, text='None', command=lambda: set_all(False)).pack(side='left', padx=4)
        tk.Button(sel, text='Uncached only', command=set_uncached).pack(side='left')

        status = tk.StringVar(value='')
        tk.Label(win, textvariable=status).pack(padx=10, pady=(4, 0))

        btns = tk.Frame(win)
        btns.pack(fill='x', padx=10, pady=10)
        run_btn = tk.Button(btns, text='Run precompute', font=self.bold)
        run_btn.pack(side='left')
        tk.Button(btns, text='Close', command=win.destroy).pack(side='right')

        def run():
            chosen = [a for a, v in vars_by_acq.items() if v.get()]
            if not chosen:
                status.set('Nothing selected.')
                return
            run_btn.configure(state='disabled')
            for i, a in enumerate(chosen, 1):
                status.set(f'Precomputing Acq {a} ({i}/{len(chosen)})...')
                win.update_idletasks()
                try:
                    SWS_utils.precompute_acq_spectrograms(self.d, a)
                except Exception as e:
                    print(f'Precompute failed for Acq {a}: {e}')
            status.set(f'Done. Precomputed {len(chosen)} acquisition(s).')
            run_btn.configure(state='normal')
            self.refresh_acqs()

        run_btn.configure(command=run)

    def _save_geometry(self):
        try:
            SWS_utils.update_layout(launcher=self.root.wm_geometry())
        except Exception:
            pass

    def on_close(self):
        self._save_geometry()
        self.root.destroy()


def launch(settings_path=None):
    root = tk.Tk()
    ScoringLauncher(root, settings_path)
    root.mainloop()


if __name__ == '__main__':
    import sys
    launch(sys.argv[1] if len(sys.argv) > 1 else None)
