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
        self.update_log = tk.BooleanVar(value=True)
        tk.Checkbutton(opt, text='Update personal log after scoring',
            variable=self.update_log).grid(row=3, column=0, sticky='w', columnspan=2)
        self._update_model_cb = opt.grid_slaves(row=2, column=0)[0]

        # Scorer mode: save to savedir/scorers/<name>/ instead of the main scoring,
        # so a second person's work stays out of every analysis until reviewed.
        # Remembered per user account, so a dedicated scorer stays in scorer mode.
        saved = SWS_utils.load_layout().get('scorer') or {}
        self.scorer_on = tk.BooleanVar(value=bool(saved.get('on', False)))
        self.scorer_name = tk.StringVar(value=str(saved.get('name', '')))
        tk.Checkbutton(opt, text='Score as a separate scorer (sequestered) — name:',
            variable=self.scorer_on, command=self._scorer_changed).grid(
            row=4, column=0, sticky='w')
        name_entry = tk.Entry(opt, textvariable=self.scorer_name, width=16)
        name_entry.grid(row=4, column=1, sticky='w')
        name_entry.bind('<FocusOut>', lambda e: self._scorer_changed())
        name_entry.bind('<Return>', lambda e: self._scorer_changed())
        # Always visible: where a Save will write.
        self.dest_var = tk.StringVar()
        self.dest_label = tk.Label(opt, textvariable=self.dest_var, font=self.bold, anchor='w')
        self.dest_label.grid(row=5, column=0, columnspan=2, sticky='w', pady=(4, 0))

        # --- Acquisition button grid ---
        mid = tk.LabelFrame(root, text='Acquisitions  —  click one to launch', padx=8, pady=6)
        mid.pack(fill='both', expand=True, padx=10, pady=4)
        legend = tk.Frame(mid)
        legend.pack(fill='x', pady=(0, 6))
        self.scored_legend = self._legend_chip(legend, COL_SCORED, 'scored')
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
        tk.Button(bot, text='Compare scorings...', command=self.open_compare_dialog).pack(side='left')
        tk.Button(bot, text='Close', font=self.bold, command=self.on_close).pack(side='right')
        self.status = tk.StringVar(value='Select a settings JSON to begin.')
        tk.Label(bot, textvariable=self.status, anchor='w').pack(side='left', padx=10)

        root.protocol('WM_DELETE_WINDOW', self.on_close)
        self._scorer_changed()
        if settings_path:
            self.load_settings(settings_path)

    def _legend_chip(self, parent, color, text):
        f = tk.Frame(parent)
        f.pack(side='left', padx=(0, 12))
        tk.Label(f, width=2, bg=color, relief='groove').pack(side='left')
        lbl = tk.Label(f, text=text)
        lbl.pack(side='left', padx=3)
        return lbl

    # ----- scorer mode -----
    def _active_scorer(self):
        """Clean scorer name if scorer mode is on, else '' (= main scoring)."""
        return SWS_utils.clean_scorer_name(self.scorer_name.get()) if self.scorer_on.get() else ''

    def _dest_settings(self):
        """A COPY of the settings carrying the active scorer, so the settings dict
        (and the JSON it came from) never changes."""
        dd = dict(self.d)
        dd['_scorer'] = self._active_scorer()
        return dd

    def _scorer_changed(self):
        who = self._active_scorer()
        if self.scorer_on.get() and not who:
            self.dest_var.set('Saving to: — enter a scorer name —')
            self.dest_label.configure(fg='#b00020')
        elif who:
            self.dest_var.set(f"Saving to: scorer '{who}' (sequestered, not used by analysis)")
            self.dest_label.configure(fg='#b77c00')
        else:
            self.dest_var.set('Saving to: MAIN scoring (used by analysis)')
            self.dest_label.configure(fg='#2e8b4f')
        # A sequestered scorer never trains the shared model.
        if who:
            self.update_model.set(False)
            self._update_model_cb.configure(state='disabled')
        else:
            self._update_model_cb.configure(state='normal')
        try:
            SWS_utils.update_layout(scorer={'on': bool(self.scorer_on.get()),
                'name': self.scorer_name.get().strip()})
        except Exception:
            pass
        if hasattr(self, 'scored_legend'):
            self.scored_legend.configure(text=f"scored by {who}" if who else 'scored (main)')
        if self.d is not None:
            self.refresh_acqs()

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
        scored = set(_scored_acquisitions(self._dest_settings()))
        cached = set(a for a in self.d.get('Acquisition', [])
                     if SWS_utils.spectrogram_cached(self.d, a))
        # ZT start of each acquisition, e.g. "ZT 10.4 +1d" = one day after the
        # first acquisition at that ZT. Cached after the first read (~3 s cold).
        self.status.set('Reading acquisition start times...')
        self.root.update_idletasks()
        try:
            zt = SWS_utils.acq_zeit_labels(self.d)
        except Exception as e:
            print(f'Could not compute ZT labels: {e}')
            zt = {}
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self._acq_order = list(self.d.get('Acquisition', []))
        for i, a in enumerate(self._acq_order):
            # tk.Label, NOT tk.Button: on macOS a Button is a native Aqua control
            # that ignores -background, so the scored/cached color coding rendered
            # as uniform grey there (the legend chips above are Labels, which is
            # why they stayed correct). Labels are drawn by Tk and honor the color
            # everywhere, so they get click and hover bindings instead of -command.
            color = self._acq_color(a, scored, cached)
            text = f'{a}\n{zt[a][2]}' if a in zt else str(a)
            b = tk.Label(self.grid_frame, text=text, width=11, bg=color,
                relief='raised', bd=2, padx=2, pady=3, cursor='hand2')
            b.grid(row=i // GRID_COLS, column=i % GRID_COLS, padx=2, pady=2, sticky='nsew')
            b.bind('<Button-1>', lambda e, a=a: self.launch_acq(a))
            b.bind('<Enter>', lambda e, w=b: w.configure(bg='#ffe08a', relief='solid'))
            b.bind('<Leave>', lambda e, w=b, c=color: w.configure(bg=c, relief='raised'))
        n_scored = len(scored & set(self._acq_order))
        n_cached = len(cached & set(self._acq_order))
        self.status.set(f'{len(self._acq_order)} acquisitions · {n_scored} scored · {n_cached} cached.')

    # ----- launch -----
    def launch_acq(self, a, start_from=None):
        """Score Acq a. start_from=<scorer> opens that scorer's version for review
        and saves to the MAIN scoring."""
        if self.d is None:
            return
        if self.scorer_on.get() and not self._active_scorer() and not start_from:
            self.status.set('Enter a scorer name (or untick scorer mode) before scoring.')
            return
        dd = self._dest_settings()
        if start_from:
            dd['_scorer'] = ''          # a review always saves to the main scoring
        self.status.set(f'Scoring Acq {a}... (see terminal)')
        self.root.update_idletasks()
        self._save_geometry()  # save now in case scoring crashes
        self.root.withdraw()
        try:
            from neuroscience_sleep_scoring import New_SWS
            New_SWS.score_acquisition(dd, a,
                use_model=self.use_model.get(), mode=self.mode.get(),
                update_model_after=self.update_model.get() and not dd['_scorer'],
                update_log_after=self.update_log.get(), start_from=start_from)
            done = f'Done with Acq {a}.'
        except Exception as e:
            import traceback
            traceback.print_exc()
            done = f'Error scoring Acq {a}: {e}'   # was overwritten by "Done" before
        finally:
            self.root.deiconify()
            self.refresh_acqs()
        self.status.set(done)

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
            # Same Aqua limitation as the grid above: a Checkbutton ignores
            # -background on macOS. Wrap it in a Frame, which Tk draws itself, so
            # the cached/uncached color still reads on every platform.
            cell_bg = COL_CACHED if a in cached else COL_PLAIN
            cell = tk.Frame(grid, bg=cell_bg, padx=1, pady=1)
            cell.grid(row=i // GRID_COLS, column=i % GRID_COLS, padx=1, pady=1, sticky='w')
            cb = tk.Checkbutton(cell, text=str(a), variable=v, width=4, bg=cell_bg)
            cb.pack()

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

    # ----- compare scorings -----
    def _sources(self):
        return ['main'] + SWS_utils.list_scorers(self.d)

    def _states_file(self, source, a, h):
        return SWS_utils.states_path(self.d, a, h, scorer='' if source == 'main' else source)

    def compare_rows(self, src_a, src_b):
        """One row per acquisition-hour scored by BOTH sources, plus pooled stats."""
        import glob, numpy as np
        rows, allA, allB = [], [], []
        da = os.path.dirname(self._states_file(src_a, 0, 0))
        for a in self.d.get('Acquisition', []):
            for fa in sorted(glob.glob(os.path.join(da, f'StatesAcq{a}_hr*.npy'))):
                h = fa.rsplit('_hr', 1)[1][:-4]
                fb = self._states_file(src_b, a, h)
                if not os.path.exists(fb):
                    continue
                A, B = np.load(fa), np.load(fb)
                r = SWS_utils.compare_states(A, B)
                r.update(acq=a, hr=h, A=A, B=B)
                rows.append(r); allA.append(A[:min(len(A), len(B))]); allB.append(B[:min(len(A), len(B))])
        pooled = SWS_utils.compare_states(np.concatenate(allA), np.concatenate(allB)) if rows else None
        return rows, pooled

    def open_compare_dialog(self):
        if self.d is None:
            self.status.set('Load a settings JSON first.')
            return
        from tkinter import ttk
        win = tk.Toplevel(self.root)
        win.title('Compare scorings')
        win.transient(self.root)
        top = tk.Frame(win); top.pack(fill='x', padx=10, pady=8)
        srcs = self._sources()
        va = tk.StringVar(value='main')
        vb = tk.StringVar(value=srcs[1] if len(srcs) > 1 else 'main')
        tk.Label(top, text='A:', font=self.bold).pack(side='left')
        tk.OptionMenu(top, va, *srcs).pack(side='left', padx=(2, 12))
        tk.Label(top, text='B:', font=self.bold).pack(side='left')
        tk.OptionMenu(top, vb, *srcs).pack(side='left', padx=(2, 12))
        if len(srcs) == 1:
            tk.Label(top, text='No sequestered scorers yet (savedir/scorers/ is empty).',
                fg='#69748a').pack(side='left', padx=8)

        cols = ('acq', 'zt', 'n', 'agree', 'kappa', 'disagree', 'top')
        heads = ('Acq', 'ZT', 'Epochs compared', 'Agreement', 'Kappa', 'Disagreements',
            'Most common disagreement (A -> B)')
        tree = ttk.Treeview(win, columns=cols, show='headings', height=14)
        for c, h_, w in zip(cols, heads, (60, 110, 120, 90, 70, 110, 230)):
            tree.heading(c, text=h_); tree.column(c, width=w, anchor='center')
        tree.pack(fill='both', expand=True, padx=10)
        summary = tk.StringVar(value='')
        tk.Label(win, textvariable=summary, font=self.bold, anchor='w').pack(fill='x', padx=10, pady=(6, 0))
        tk.Label(win, text='Epochs unscored by either side are left out. Kappa corrects '
            'agreement for chance (1 = perfect, 0 = chance level).', fg='#69748a',
            anchor='w', justify='left').pack(fill='x', padx=10)
        state = {'rows': []}
        names = {1: 'Wake', 2: 'NREM', 3: 'REM'}

        def fmt(x, pct=False):
            import math
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return '—'
            return f'{100 * x:.1f}%' if pct else f'{x:.2f}'

        def run():
            tree.delete(*tree.get_children())
            if va.get() == vb.get():
                summary.set('Pick two different sources.'); return
            rows, pooled = self.compare_rows(va.get(), vb.get())
            state['rows'] = rows
            try:
                zt = SWS_utils.acq_zeit_labels(self.d)
            except Exception:
                zt = {}
            for i, r in enumerate(rows):
                conf = r['confusion'].copy()
                for k in range(len(conf)):
                    conf[k, k] = 0
                top_ = ''
                if conf.sum():
                    ii, jj = divmod(int(conf.argmax()), conf.shape[1])
                    top_ = f"{names[r['states'][ii]]} -> {names[r['states'][jj]]}: {conf[ii, jj]}"
                tree.insert('', 'end', iid=str(i), values=(
                    r['acq'] if r['hr'] == '0' else f"{r['acq']} hr{r['hr']}",
                    zt.get(r['acq'], (None, None, '—'))[2], r['n_compared'],
                    fmt(r['agreement'], True), fmt(r['kappa']), len(r['disagree_idx']), top_))
            if pooled:
                summary.set(f"{va.get()} vs {vb.get()}: {len(rows)} acquisition-hour(s) scored by both "
                    f"· overall agreement {fmt(pooled['agreement'], True)} · kappa {fmt(pooled['kappa'])}")
            else:
                summary.set(f'No acquisitions scored by both {va.get()} and {vb.get()}.')

        def selected():
            sel = tree.selection()
            return state['rows'][int(sel[0])] if sel else None

        def show():
            r = selected()
            if r is None:
                summary.set('Select an acquisition first.'); return
            self._show_comparison_figure(r, va.get(), vb.get())

        def review():
            r = selected()
            if r is None:
                summary.set('Select an acquisition first.'); return
            if vb.get() == 'main':
                summary.set('Set B to the scorer you want to review.'); return
            win.destroy()
            self.launch_acq(r['acq'], start_from=vb.get())

        btns = tk.Frame(win); btns.pack(fill='x', padx=10, pady=8)
        tk.Button(btns, text='Compare', font=self.bold, command=run).pack(side='left')
        tk.Button(btns, text='Show selected', command=show).pack(side='left', padx=6)
        tk.Button(btns, text="Review B in the scoring GUI (saves to main)...",
            command=review).pack(side='left')
        tk.Button(btns, text='Close', command=win.destroy).pack(side='right')
        tree.bind('<Double-1>', lambda e: show())
        run()

    def _show_comparison_figure(self, r, name_a, name_b):
        """Both hypnograms, where they disagree, and the confusion matrix, drawn
        with Tk's own matplotlib canvas so it never competes with the scoring
        GUI's event loop."""
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.colors import ListedColormap
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        A = np.asarray(r['A'], float); B = np.asarray(r['B'], float)
        n = min(len(A), len(B)); A, B = A[:n], B[:n]
        cmap = ListedColormap([SWS_utils.STATE_COLORS[k] for k in (0, 1, 2, 3)])
        clean = lambda X: np.where(np.isin(X, (1, 2, 3)), X, 0)
        fig = Figure(figsize=(11, 4.2), dpi=100)
        gs = fig.add_gridspec(3, 2, width_ratios=[4, 1.25], hspace=0.35, wspace=0.25)
        for row, (X, lab) in enumerate(((A, f'A: {name_a}'), (B, f'B: {name_b}'))):
            ax = fig.add_subplot(gs[row, 0])
            ax.imshow(clean(X)[None, :], aspect='auto', cmap=cmap, vmin=-0.5, vmax=3.5,
                interpolation='nearest', extent=(0, n, 0, 1))
            ax.set_yticks([]); ax.set_ylabel(lab, rotation=0, ha='right', va='center')
            ax.set_xticklabels([])
        axd = fig.add_subplot(gs[2, 0])
        both = np.isin(A, (1, 2, 3)) & np.isin(B, (1, 2, 3))
        dis = both & (A != B)
        axd.vlines(np.where(dis)[0] + 0.5, 0, 1, color='k', lw=0.8)
        axd.set_xlim(0, n); axd.set_ylim(0, 1); axd.set_yticks([])
        axd.set_ylabel('disagree', rotation=0, ha='right', va='center')
        axd.set_xlabel('bin')
        axc = fig.add_subplot(gs[:, 1])
        conf = r['confusion']; lab = ['Wake', 'NREM', 'REM']
        axc.imshow(conf, cmap='Greys')
        for i in range(3):
            for j in range(3):
                axc.text(j, i, int(conf[i, j]), ha='center', va='center',
                    color='white' if conf[i, j] > conf.max() / 2 else 'black', fontsize=9)
        axc.set_xticks(range(3)); axc.set_xticklabels(lab, fontsize=8)
        axc.set_yticks(range(3)); axc.set_yticklabels(lab, fontsize=8)
        axc.set_xlabel(f'B: {name_b}'); axc.set_ylabel(f'A: {name_a}')
        k = r['kappa']
        fig.suptitle(f"Acq {r['acq']} hr {r['hr']} — agreement {100 * r['agreement']:.1f}% · "
            f"kappa {k:.2f} · {len(r['disagree_idx'])} disagreeing bins", fontweight='bold')
        top = tk.Toplevel(self.root)
        top.title(f"Acq {r['acq']}: {name_a} vs {name_b}")
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        return fig

    def _save_geometry(self):
        try:
            SWS_utils.update_layout(launcher=self.root.wm_geometry())
        except Exception:
            pass

    def on_close(self):
        """Save layout and fully exit the program.

        A plain root.destroy() sometimes left the process alive because scoring
        leaves stray Tk roots (the state-selection popup) and matplotlib figure
        windows around. Tear those down and hard-exit so Close reliably quits.
        """
        self._save_geometry()
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
        try:
            from neuroscience_sleep_scoring import New_SWS
            # The reusable state-selection popup is cached between corrections.
            New_SWS.destroy_state_popup()
            if getattr(New_SWS, '_state_popup_root', None) is not None:
                New_SWS._state_popup_root.destroy()
                New_SWS._state_popup_root = None
        except Exception:
            pass
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)


def launch(settings_path=None):
    root = tk.Tk()
    ScoringLauncher(root, settings_path)
    root.mainloop()


if __name__ == '__main__':
    import sys
    launch(sys.argv[1] if len(sys.argv) > 1 else None)
