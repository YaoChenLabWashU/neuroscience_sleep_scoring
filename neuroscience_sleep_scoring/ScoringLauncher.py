"""Central tkinter launcher for the sleep-scoring GUI.

Replaces the chain of terminal input() prompts: pick a Score_Settings JSON, see
the acquisitions (green = already scored), set the usual choices as checkboxes,
and press "Launch GUI" to score one. The scoring figures reopen in their saved
positions (see SWS_utils layout helpers) with the video already at frame 0.

stdout still carries all the scoring prints for debugging.
"""
import os
import json
import tkinter as tk
from tkinter import filedialog, font as tkfont

from neuroscience_sleep_scoring import SWS_utils


def _scored_acquisitions(d):
    """Sorted acquisition numbers that already have a State file. Lazy-imports
    New_SWS to avoid a circular import at module load."""
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

        # Restore launcher window geometry from the shared layout store.
        geo = SWS_utils.load_layout().get('launcher')
        if geo:
            try:
                root.wm_geometry(geo)
            except Exception:
                pass

        bold = tkfont.Font(weight='bold')

        # --- Settings file row ---
        top = tk.Frame(root)
        top.pack(fill='x', padx=10, pady=(10, 4))
        tk.Label(top, text='Settings JSON:', font=bold).pack(side='left')
        self.path_var = tk.StringVar(value='(none selected)')
        tk.Label(top, textvariable=self.path_var, anchor='w').pack(
            side='left', fill='x', expand=True, padx=6)
        tk.Button(top, text='Browse...', command=self.browse).pack(side='right')

        # --- Options ---
        opt = tk.LabelFrame(root, text='Options', padx=8, pady=6)
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

        # --- Acquisition list ---
        mid = tk.LabelFrame(root, text='Acquisitions (green = already scored)',
            padx=8, pady=6)
        mid.pack(fill='both', expand=True, padx=10, pady=4)
        listframe = tk.Frame(mid)
        listframe.pack(fill='both', expand=True)
        self.listbox = tk.Listbox(listframe, height=12, exportselection=False)
        sb = tk.Scrollbar(listframe, orient='vertical', command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self.listbox.bind('<Double-Button-1>', lambda e: self.launch_selected())

        # --- Launch + status ---
        bot = tk.Frame(root)
        bot.pack(fill='x', padx=10, pady=(4, 10))
        self.launch_btn = tk.Button(bot, text='Launch GUI', font=bold,
            command=self.launch_selected, state='disabled')
        self.launch_btn.pack(side='left')
        tk.Button(bot, text='Refresh', command=self.refresh_acqs).pack(side='left', padx=6)
        self.status = tk.StringVar(value='Select a settings JSON to begin.')
        tk.Label(bot, textvariable=self.status, anchor='e').pack(side='right')

        root.protocol('WM_DELETE_WINDOW', self.on_close)
        if settings_path:
            self.load_settings(settings_path)

    # ----- settings / acquisition list -----
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
        self.launch_btn.configure(state='normal')

    def refresh_acqs(self):
        if self.d is None:
            return
        scored = set(_scored_acquisitions(self.d))
        self.listbox.delete(0, 'end')
        self._acq_order = list(self.d.get('Acquisition', []))
        for i, a in enumerate(self._acq_order):
            label = f'Acq {a}' + ('   ✓ scored' if a in scored else '')
            self.listbox.insert('end', label)
            if a in scored:
                self.listbox.itemconfig(i, foreground='green')
        n_scored = len(scored & set(self._acq_order))
        self.status.set(f'{len(self._acq_order)} acquisitions, {n_scored} scored.')

    # ----- launch -----
    def launch_selected(self):
        if self.d is None:
            return
        sel = self.listbox.curselection()
        if not sel:
            self.status.set('Pick an acquisition first.')
            return
        a = self._acq_order[sel[0]]
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
