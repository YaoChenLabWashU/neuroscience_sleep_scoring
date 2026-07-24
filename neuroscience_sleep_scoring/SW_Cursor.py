import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkinter as tk
from tkinter import font as tkfont

DEBUG = False

class Cursor(object):
    def __init__(self, ax1, ax2, ax4, all_axes=None, epochlen=4, fig2_axes=None):
        self.clicked=False
        self.second_click = False
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax4 = ax4
        self.all_axes = all_axes if all_axes else [ax1, ax2, ax4]
        self.fig2_axes = fig2_axes if fig2_axes else []  # ax6-ax10
        self.movie_mode = False
        self.epochlen = epochlen
        self.bins = []
        self.change_bins = False
        self.movie_bin = 0
        self.DONE = False
        self.STATE = []
        self.popup_xy = None

        # Replot flags
        self.replot = False
        self.replotx = 0
        
        # Current position in seconds for crosshair sync
        self.current_x_sec = 0

        # Video window with slider (p key opens)
        self.video_cap = None
        self.video_timestamp = None
        self.video_d = None
        self.preview_window_name = 'Frame'
        self.preview_window_open = False
        self._preview_visible = False  # Track whether window is currently visible
        self.last_preview_frame_idx = None
        self._video_slider_max = 1000
        self._video_max_time = 0
        
        # Magnify mode ('g' key) - show zoomed view around cursor
        self.magnify_mode = False
        self.magnify_callback = None  # Callback to update magnified view
        self.magnify_half_window = 90  # ±seconds for magnify view (adjustable via 'v')
        self.magnify_emg_ylim = None

        # When set (by 'm' microarousal), the main loop assigns this state to the
        # selected bin(s) and skips the state-selection popup. Cleared after use.
        self.forced_state = None

        # Microarousal placement mode: press 'm' to arm (and resize 1-4 bins), then
        # click the scoring plot to drop a Wake block of that width.
        self.micro_mode = False
        self.micro_size = 1

        # 'l' reference lines (dashed), and the configurable ±s x-span of the
        # detailed overview spectrograms (View Settings, 'v').
        self._ref_lines = []
        self.detail_spect_halfspan = 600

        # Cached video file reference to reduce lookup overhead
        self._cached_video_filename = None
        self._cached_video_cap_ref = None

        # Help window reference (persistent, non-modal)
        self._help_window = None

        # Recursion guard for magnify/preview callbacks
        self._in_magnify_callback = False

        # Blitting support
        self.background = None
        self.background_fig2 = None

        # Create vertical lines for ALL fig1 axes (for full-height crosshair)
        self.vertical_lines = []
        for ax in self.all_axes:
            vline = ax.axvline(color='k', lw=0.8, ls='--', animated=True)
            self.vertical_lines.append(vline)
        
        # Create vertical lines for fig2 axes
        self.vertical_lines_fig2 = []
        for ax in self.fig2_axes:
            vline = ax.axvline(color='k', lw=0.8, ls='--', animated=True)
            self.vertical_lines_fig2.append(vline)
        
        # Horizontal line and text only in ax2
        self.horizontal_line = ax2.axhline(color='k', lw=0.8, ls='--', animated=True)
        self.text = ax2.text(0.72, 0.9, '', transform=ax2.transAxes, animated=True)

        # initializing the marker lines (non-animated, for epoch marking)
        self.ylims_ax1 = ax1.get_ylim()
        self.ylims_ax2 = ax2.get_ylim()

        line1 = ax1.plot([0,0], [self.ylims_ax1[0], self.ylims_ax1[1]], linewidth = 0.5, color = 'k')
        ml1 = line1.pop(0)

        line2 = ax2.plot([0,0], [self.ylims_ax2[0], self.ylims_ax2[1]], linewidth = 0.5, color = 'k')
        ml2 = line2.pop(0)
        
        # Current epoch marker line in ax2 (moved with 'c' key)
        # ANIMATED so it can be blitted for speed
        self.current_epoch_t = 0
        self.epoch_marker = ax2.axvline(0, linewidth=1.5, color='darkblue', animated=True)

        # Separate recursion guard for video slider (don't reuse _in_magnify_callback)
        self._in_slider_callback = False

        self.movement_x_axis = np.linspace(0,60,900)
        self.spect_x_axis = np.linspace(199,1442, 900)

        self.lines = [ml1, ml2]
        self.toggle_line = False

        # Recapture the blit background after every full redraw (draw_event) so the
        # crosshair never has to force its own slow canvas.draw() on mouse move and
        # never vanishes after a correction/replot. This is the key to a fast,
        # stable crosshair; on_mouse_move only restores + blits, never draws.
        try:
            self.ax2.figure.canvas.mpl_connect('draw_event', self._on_draw)
        except Exception:
            pass
        if self.fig2_axes:
            try:
                self.fig2_axes[0].figure.canvas.mpl_connect('draw_event', self._on_draw_fig2)
            except Exception:
                pass

        if DEBUG: print('making a cursor')

    def _on_draw(self, event):
        """After any full redraw, recapture the clean background AND immediately
        re-blit the crosshair, so it persists (never blinks out) across
        corrections, replots and resizes."""
        try:
            self.background = self.ax2.figure.canvas.copy_from_bbox(self.ax2.figure.bbox)
        except Exception:
            self.background = None
            return
        self._redraw_crosshair()

    def _on_draw_fig2(self, event):
        """Cache fig2 as a bitmap after any full redraw (for the magnify crosshair)."""
        if self.fig2_axes:
            try:
                self.background_fig2 = self.fig2_axes[0].figure.canvas.copy_from_bbox(
                    self.fig2_axes[0].figure.bbox)
            except Exception:
                self.background_fig2 = None

    def on_resize(self, event):
        """Invalidate blitting background on window resize (draw_event refreshes it)."""
        self.background = None

    def on_resize_fig2(self, event):
        """Invalidate blitting background on fig2 resize (draw_event refreshes it)."""
        self.background_fig2 = None

    def on_move(self, event):
        if DEBUG: print('on move')

    def on_press(self, event):
        if event.key == 'd':
            if DEBUG: print('DONE SCORING')
            self.DONE = True
        elif event.key == 'p':
            # Play the current bin's video once (same as 'o').
            if self.video_cap is not None and self.video_timestamp is not None:
                self._play_current_bin()
            else:
                print('No video available')
        elif event.key == 'm':
            # Microarousal placement: press 'm' to arm placement mode; press again
            # to grow the block width 1->2->3->4->1. Then CLICK on the scoring plot
            # (ax2) to drop that many Wake bins starting there (handled in on_click).
            if not self.micro_mode:
                self.micro_mode = True
                self.micro_size = 1
            else:
                self.micro_size = self.micro_size % 4 + 1
            print(f'Microarousal armed: {self.micro_size} bin(s) of Wake. '
                'Press m to resize, click the scoring plot to place, r/esc to cancel.')
        elif event.key == 'g':
            # Toggle magnify mode (live zoomed view that follows the cursor)
            self.magnify_mode = not self.magnify_mode
            if self.magnify_mode:
                print(f'Magnify mode ON - drag cursor to see \u00b1{self.magnify_half_window}s zoomed view')
            else:
                print('Magnify mode OFF')
        elif event.key == 'v':
            # Show view-settings popup
            self._show_view_settings()
        elif event.key == 'c':
            # Move current bin marker to cursor position
            bin_idx = int(self.current_x_sec // self.epochlen)
            self.epoch_marker.set_xdata([bin_idx, bin_idx])
            self.current_epoch_t = self.current_x_sec
            if DEBUG: print(f'Moved current bin marker to bin {bin_idx}')
        elif event.key == 'o':
            # Play the current bin's video once.
            if self.video_cap is not None and self.video_timestamp is not None:
                self._play_current_bin()
            else:
                print('No video available')
        elif event.key == 'i':
            # Show quick reference GUI (persistent, non-modal)
            self._show_help_popup()
        elif event.key in ('r', 'escape'):
            # Cancel a pending click selection or armed microarousal placement.
            if self.micro_mode:
                self.micro_mode = False
                print('Microarousal placement cancelled')
            if self.clicked:
                self.clicked = False
                self.bins = []
                print('Selection cancelled')
        elif event.key == 'left':
            if self.clicked and len(self.bins) == 1:
                self.bins[0] = max(0, self.bins[0] - 1)
                print(f'Selection start: bin {self.bins[0]}')
        elif event.key == 'right':
            if self.clicked and len(self.bins) == 1:
                self.bins[0] = self.bins[0] + 1
                print(f'Selection start: bin {self.bins[0]}')
        elif event.key in [1,2,3,4]:
            self.STATE.append(event.key)
        elif event.key == 'l':
            # Toggle a dashed reference line across all fig1 panels at the mouse x.
            # (Does not play video and never errors when the cursor is off-axis.)
            if getattr(self, '_ref_lines', None):
                for ln in self._ref_lines:
                    try:
                        ln.remove()
                    except Exception:
                        pass
                self._ref_lines = []
                self.background = None
                self.ax2.figure.canvas.draw_idle()
            elif event.inaxes in self.all_axes and event.xdata is not None:
                x_sec = self._event_x_to_seconds(event)
                self._ref_lines = []
                for ax in self.all_axes:
                    x = self._x_for_axis(ax, x_sec)
                    self._ref_lines.append(ax.axvline(x, color='k', lw=1, ls='--'))
                self.background = None
                self.ax2.figure.canvas.draw_idle()
            else:
                print("Hover over a plot, then press 'l' to place a reference line.")

    def _show_view_settings(self):
        """Popup ('v') to adjust view parameters: the detailed overview-spectrogram
        x-span, the magnify window, and the EMG y-limits. Shares matplotlib's Tk
        root (a separate tk.Tk() + mainloop() can deadlock the GUI)."""
        try:
            root = getattr(tk, '_default_root', None)
            temp = None
            if root is None:
                root = tk.Tk()
                root.withdraw()
                temp = root

            win = tk.Toplevel(root)
            win.title('View Settings')
            win.resizable(False, False)
            win.attributes('-topmost', True)
            bold_font = tkfont.Font(weight='bold')
            tk.Label(win, text='View Settings', font=bold_font).pack(padx=10, pady=6)

            # Detailed overview-spectrogram x-span (fig2 ax6/ax7).
            tk.Label(win, text='Detailed spectrogram x-span (±seconds):').pack(padx=10, pady=(6, 2))
            var_span = tk.IntVar(value=int(self.detail_spect_halfspan))
            tk.Scale(win, from_=30, to=1800, resolution=10, orient='horizontal',
                variable=var_span, length=230).pack(padx=10, pady=2)

            # Magnify window (±seconds).
            tk.Label(win, text='Magnify window (±seconds):').pack(padx=10, pady=(6, 2))
            var_window = tk.IntVar(value=self.magnify_half_window)
            tk.Scale(win, from_=10, to=300, orient='horizontal',
                variable=var_window, length=230).pack(padx=10, pady=2)

            # EMG y-limits (blank = auto).
            tk.Label(win, text='EMG y-limits (min, max; blank = auto):').pack(padx=10, pady=(6, 2))
            emg_frame = tk.Frame(win)
            emg_frame.pack(padx=10, pady=2)
            emg_min_var = tk.StringVar(value='' if self.magnify_emg_ylim is None else str(self.magnify_emg_ylim[0]))
            emg_max_var = tk.StringVar(value='' if self.magnify_emg_ylim is None else str(self.magnify_emg_ylim[1]))
            tk.Entry(emg_frame, textvariable=emg_min_var, width=8).pack(side='left', padx=4)
            tk.Entry(emg_frame, textvariable=emg_max_var, width=8).pack(side='left', padx=4)

            def apply_settings():
                self.detail_spect_halfspan = var_span.get()
                self.magnify_half_window = var_window.get()
                emg_min = emg_min_var.get().strip()
                emg_max = emg_max_var.get().strip()
                if emg_min and emg_max:
                    try:
                        self.magnify_emg_ylim = (float(emg_min), float(emg_max))
                    except ValueError:
                        self.magnify_emg_ylim = None
                else:
                    self.magnify_emg_ylim = None
                if self.magnify_emg_ylim is not None:
                    self.ax4.set_ylim(self.magnify_emg_ylim)
                    if len(self.fig2_axes) > 4:
                        self.fig2_axes[4].set_ylim(self.magnify_emg_ylim)
                # Apply the detailed spectrogram x-span now, centered on the current epoch.
                if len(self.fig2_axes) >= 2:
                    c = self.current_epoch_t
                    self.fig2_axes[0].set_xlim([c - self.detail_spect_halfspan, c + self.detail_spect_halfspan])
                    self.fig2_axes[1].set_xlim([c - self.detail_spect_halfspan, c + self.detail_spect_halfspan])
                self.background = None
                self.background_fig2 = None
                try:
                    self.ax4.figure.canvas.draw_idle()
                except Exception:
                    pass
                if len(self.fig2_axes) > 0:
                    try:
                        self.fig2_axes[0].figure.canvas.draw_idle()
                    except Exception:
                        pass
                print(f'View settings: detailed spectrogram ±{self.detail_spect_halfspan}s, '
                    f'magnify ±{self.magnify_half_window}s')
                win.destroy()

            tk.Button(win, text='Apply', command=apply_settings, width=12).pack(padx=10, pady=8)
            try:
                win.update_idletasks()
                win.wait_visibility()
                win.grab_set()
            except Exception:
                pass
            root.wait_window(win)
            if temp is not None:
                temp.destroy()
        except Exception as e:
            print(f'View settings popup error: {e}')

    def _show_help_popup(self):
        """Show keyboard shortcuts reference as a blocking, closeable window."""
        try:
            root = tk.Tk()
            root.title('Keyboard Shortcuts')
            root.resizable(False, False)
            root.attributes('-topmost', True)

            bold_font = tkfont.Font(weight='bold')

            shortcuts = [
                ('click x2', 'Select start/end bins, then pick state'),
                ('m', 'Arm microarousal (press again: 1->4 bins), then click to place Wake'),
                ('d', 'Done scoring (save and close)'),
                ('o', 'Play 4s video clip at cursor'),
                ('p', 'Toggle video window visibility'),
                ('g', 'Toggle magnify mode'),
                ('v', 'View settings (detail spectrogram x-span, magnify, EMG y-lim)'),
                ('l', 'Toggle a dashed reference line at the cursor'),
                ('c', 'Move current bin marker to cursor'),
                ('←/→', 'Nudge selection start (after 1st click)'),
                ('r', 'Cancel current selection'),
                ('1/2/3/4', 'Set state (Wake/NREM/REM/Other)'),
                ('i', 'Show this help'),
            ]

            tk.Label(root, text='Keyboard Shortcuts', font=bold_font).pack(padx=12, pady=8)

            frame = tk.Frame(root)
            frame.pack(padx=12, pady=4)
            for key, desc in shortcuts:
                row = tk.Frame(frame)
                row.pack(fill='x', pady=1)
                tk.Label(row, text=key, font=bold_font, width=8, anchor='e').pack(side='left')
                tk.Label(row, text=f'  {desc}', anchor='w').pack(side='left')

            def on_close():
                root.destroy()

            root.protocol('WM_DELETE_WINDOW', on_close)
            tk.Button(root, text='Close', command=on_close, width=10).pack(pady=8)
            root.mainloop()
        except Exception as e:
            if DEBUG: print(f'Help popup error: {e}')


    def _video_frame_size(self):
        """Native (width, height) of the behavior video, so the preview window can
        match the frame size. Returns (None, None) if it can't be determined."""
        try:
            if self.video_cap:
                for c in self.video_cap.values():
                    if c is not None and c.isOpened():
                        w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if w > 0 and h > 0:
                            return w, h
        except Exception:
            pass
        return None, None

    def _draw_banner(self, frame, time_sec, frame_idx):
        """Draw the time / bin / frame banner with a transparent background.

        Uses outlined text (thick black stroke under a thin yellow stroke) so it's
        legible over any frame content without a filled box occluding the video."""
        bin_idx = int(time_sec // self.epochlen)
        label = f"t={time_sec:.2f}s   bin={bin_idx}   frame={frame_idx}"
        cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _play_current_bin(self):
        """Play the video of the bin under the cursor exactly once (no loop).

        Opens the preview window (at the video's native frame size) if needed,
        plays the frames spanning this bin's epoch a single time, and leaves the
        last frame on screen. Bound to both 'o' and 'p'. Errors are printed."""
        if self.video_timestamp is None or self.video_cap is None:
            print('No video available')
            return
        try:
            epochlen = self.epochlen
            # Snap to the start of the bin under the cursor.
            time_sec = math.floor(max(self.current_x_sec, 0) / epochlen) * epochlen
            offset_times = self.video_timestamp['Offset_Time'].values
            clip_end = time_sec + epochlen

            start_pos = max(np.searchsorted(offset_times, time_sec, side='right') - 1, 0)
            end_pos = min(np.searchsorted(offset_times, clip_end, side='right') - 1,
                len(offset_times) - 1)

            start_frame = int(self.video_timestamp.index[start_pos])
            end_frame = int(self.video_timestamp.index[end_pos])
            if end_frame < start_frame:
                print('No video frames for this bin.')
                return

            csv_filename = self.video_timestamp['Filename'][start_frame]
            from neuroscience_sleep_scoring import SWS_utils
            v = SWS_utils.get_videofn_from_csv(self.video_d, csv_filename)
            if v not in self.video_cap or not self.video_cap[v].isOpened():
                print('Cannot open video file: ' + str(v))
                return
            cap = self.video_cap[v]

            if not self.preview_window_open:
                self._ensure_preview_window()
            elif not self._preview_visible:
                self._show_preview_window()

            print(f'Playing bin {int(time_sec//epochlen)} '
                f'(t={time_sec:.1f}-{clip_end:.1f}s)')
            index_vals = self.video_timestamp.index.values
            shown = 0
            last_t = time_sec
            for f_idx in range(start_frame, end_frame + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                shown += 1
                if f_idx in self.video_timestamp.index:
                    last_t = float(self.video_timestamp.loc[f_idx, 'Offset_Time'])
                else:
                    pos = min(np.searchsorted(index_vals, f_idx), len(offset_times) - 1)
                    last_t = float(offset_times[pos])
                cv2.imshow(self.preview_window_name, self._draw_banner(frame, last_t, f_idx))
                if (cv2.waitKey(30) & 0xFF) in (ord('q'), 27):
                    break
            if shown == 0:
                print('Could not read video frames for this bin.')
                return
            # Leave the last frame of the bin on screen.
            self.last_preview_frame_idx = None
            self._show_preview_frame(last_t)
        except Exception as e:
            print(f'Video playback error: {e}')

    def _x_for_axis(self, ax, x):
        if ax == self.ax2:
            return x / self.epochlen
        return x

    def _event_x_to_seconds(self, event):
        # Handle fig1 axes
        if event.inaxes == self.ax2:
            return event.xdata * self.epochlen
        if event.inaxes in self.all_axes:
            return event.xdata
        # Handle fig2 axes - they use seconds directly
        if event.inaxes in self.fig2_axes:
            return event.xdata
        return event.xdata

    def _set_cursor_x(self, x, y=None):
        if y is not None:
            self.horizontal_line.set_ydata([y, y])
        for i, vline in enumerate(self.vertical_lines):
            ax = self.all_axes[i]
            x_mapped = self._x_for_axis(ax, x)
            vline.set_xdata([x_mapped, x_mapped])
        # Update fig2 vertical lines (fig2 uses seconds directly)
        for vline in self.vertical_lines_fig2:
            vline.set_xdata([x, x])

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        for vline in self.vertical_lines:
            vline.set_visible(visible)
        for vline in self.vertical_lines_fig2:
            vline.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move_fig2(self, event):
        """Handle mouse movement on fig2 (detailed view). Only active in magnify mode."""
        if not self.magnify_mode:
            return
        if not event.inaxes or event.inaxes not in self.fig2_axes:
            return
        
        # Initialize background for blitting on first call
        if self.background_fig2 is None and len(self.fig2_axes) > 0:
            fig2 = self.fig2_axes[0].figure
            fig2.canvas.draw()
            self.background_fig2 = fig2.canvas.copy_from_bbox(fig2.bbox)
        
        x_sec = event.xdata
        self.current_x_sec = x_sec
        
        # Update fig2 lines
        for vline in self.vertical_lines_fig2:
            vline.set_xdata([x_sec, x_sec])
        
        # Blit fig2
        if self.background_fig2 is not None and len(self.fig2_axes) > 0:
            fig2 = self.fig2_axes[0].figure
            fig2.canvas.restore_region(self.background_fig2)
            for i, vline in enumerate(self.vertical_lines_fig2):
                self.fig2_axes[i].draw_artist(vline)
            fig2.canvas.blit(fig2.bbox)
        
        # Also update fig1 crosshair
        self._set_cursor_x(x_sec)
        if self.background is not None:
            self.ax2.figure.canvas.restore_region(self.background)
            self.ax2.draw_artist(self.horizontal_line)
            self.ax2.draw_artist(self.epoch_marker)
            for i, vline in enumerate(self.vertical_lines):
                self.all_axes[i].draw_artist(vline)
            self.ax2.draw_artist(self.text)
            self.ax2.figure.canvas.blit(self.ax2.figure.bbox)
        
        # Magnify mode callback - follows fast cursor on fig2
        if self.magnify_callback is not None and not self._in_magnify_callback:
            self._in_magnify_callback = True
            try:
                self.magnify_callback(x_sec)
            finally:
                self._in_magnify_callback = False

    def _redraw_crosshair(self):
        """Restore the clean background and blit the crosshair at its last position.

        Called on every mouse move AND after every full redraw (_on_draw), so the
        crosshair is always visible (never disappears, even while stationary after
        a correction) and never has to force a slow full draw. The vertical lines
        stay visible across all panels; only the horizontal line follows the y of
        whichever panel the cursor is over."""
        fig = self.ax2.figure
        if self.background is None:
            try:
                self.background = fig.canvas.copy_from_bbox(fig.bbox)
            except Exception:
                return
        fig.canvas.restore_region(self.background)
        for i, vline in enumerate(self.vertical_lines):
            self.all_axes[i].draw_artist(vline)
        self.ax2.draw_artist(self.epoch_marker)
        if self.horizontal_line.get_visible():
            self.ax2.draw_artist(self.horizontal_line)
        self.ax2.draw_artist(self.text)
        fig.canvas.blit(fig.bbox)

    def on_mouse_move(self, event):
        if event.inaxes:
            x_sec = self._event_x_to_seconds(event)
            self.current_x_sec = x_sec
            self._set_cursor_x(x_sec, y=event.ydata)
            self.text.set_text('x=%1.2f, y=%1.2f' % (x_sec, event.ydata))
            self.horizontal_line.set_visible(True)
        else:
            # Off-axis: leave the vertical line where it was (persistent) and just
            # drop the horizontal line (only meaningful over a panel).
            self.horizontal_line.set_visible(False)
        self._redraw_crosshair()

        # Sync the fig2 (detail) crosshair + zoom only in magnify mode.
        if event.inaxes and self.magnify_mode and len(self.fig2_axes) > 0:
            if self.background_fig2 is None:
                self.background_fig2 = self.fig2_axes[0].figure.canvas.copy_from_bbox(
                    self.fig2_axes[0].figure.bbox)
            fig2 = self.fig2_axes[0].figure
            fig2.canvas.restore_region(self.background_fig2)
            for i, vline in enumerate(self.vertical_lines_fig2):
                self.fig2_axes[i].draw_artist(vline)
            fig2.canvas.blit(fig2.bbox)
            if self.magnify_callback is not None and not self._in_magnify_callback:
                self._in_magnify_callback = True
                try:
                    self.magnify_callback(self.current_x_sec)
                finally:
                    self._in_magnify_callback = False

    def _show_preview_frame(self, time_sec):
        """Show video frame at the given time in seconds. Only updates if window is visible."""
        if self.video_timestamp is None or self.video_cap is None:
            return
        if not self.preview_window_open or not self._preview_visible:
            return
        try:
            # Find frame index for this time using searchsorted for speed
            offset_times = self.video_timestamp['Offset_Time'].values
            pos = np.searchsorted(offset_times, time_sec, side='right') - 1
            if pos < 0:
                return
            frame_idx = self.video_timestamp.index[pos]

            if self.last_preview_frame_idx == frame_idx:
                return
            self.last_preview_frame_idx = frame_idx
            
            # Get video capture reference, using cache to skip repeated lookups
            csv_filename = self.video_timestamp['Filename'][frame_idx]
            if csv_filename != self._cached_video_filename or self._cached_video_cap_ref is None:
                from neuroscience_sleep_scoring import SWS_utils
                v = SWS_utils.get_videofn_from_csv(self.video_d, csv_filename)
                if v in self.video_cap and self.video_cap[v].isOpened():
                    self._cached_video_filename = csv_filename
                    self._cached_video_cap_ref = self.video_cap[v]
                else:
                    return
            
            cap = self._cached_video_cap_ref
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                cv2.imshow(self.preview_window_name, self._draw_banner(frame, time_sec, frame_idx))
                # Update slider position (use _in_slider_callback guard to prevent recursion)
                if self._video_max_time > 0 and self._video_slider_max > 0 and not self._in_slider_callback:
                    self._in_slider_callback = True
                    try:
                        slider_val = int((time_sec / self._video_max_time) * self._video_slider_max)
                        slider_val = max(0, min(slider_val, self._video_slider_max))
                        cv2.setTrackbarPos('Time', self.preview_window_name, slider_val)
                    finally:
                        self._in_slider_callback = False
                cv2.waitKey(1)
        except Exception as e:
            if DEBUG: print(f'Preview error: {e}')

    def _on_video_slider(self, val):
        """Callback for video slider trackbar."""
        if self._video_max_time <= 0:
            return
        # Guard against recursion: _show_preview_frame calls setTrackbarPos which calls this
        if self._in_slider_callback:
            return
        self._in_slider_callback = True
        try:
            time_sec = (val / self._video_slider_max) * self._video_max_time
            self.current_x_sec = time_sec
            self._set_cursor_x(time_sec)
            self._show_preview_frame(time_sec)
            # Update fig1 cursor via blit
            fig = self.ax2.figure
            if self.background is None:
                fig.canvas.draw()
                self.background = fig.canvas.copy_from_bbox(fig.bbox)
            try:
                fig.canvas.restore_region(self.background)
                for i, vline in enumerate(self.vertical_lines):
                    self.all_axes[i].draw_artist(vline)
                self.ax2.draw_artist(self.horizontal_line)
                self.ax2.draw_artist(self.epoch_marker)
                self.ax2.draw_artist(self.text)
                fig.canvas.blit(fig.bbox)
            except Exception:
                pass
            # Trigger magnify update if in magnify mode
            if self.magnify_mode and self.magnify_callback is not None:
                self.magnify_callback(time_sec)
        finally:
            self._in_slider_callback = False

    def _toggle_preview_window(self):
        """Toggle video window visibility. Creates once, then shows/hides."""
        if not self.preview_window_open:
            # First time: create the window
            self._ensure_preview_window()
            self._show_preview_frame(self.current_x_sec)
            print('Video window opened - use slider or arrows to navigate')
        elif self._preview_visible:
            # Hide the window - save geometry first
            self._update_preview_window_props()
            try:
                cv2.setWindowProperty(self.preview_window_name, cv2.WND_PROP_VISIBLE, 0)
            except Exception:
                pass
            self._preview_visible = False
            print('Video window hidden (press p or e to show)')
        else:
            # Show the window again at its existing position/size
            self._show_preview_window()
            self.last_preview_frame_idx = None  # Force frame refresh
            self._show_preview_frame(self.current_x_sec)
            print('Video window shown')

    def _show_preview_window(self):
        """Make the existing preview window visible again without changing its size/position."""
        try:
            cv2.setWindowProperty(self.preview_window_name, cv2.WND_PROP_VISIBLE, 1)
        except Exception:
            # If setWindowProperty doesn't work, recreate the window
            self.preview_window_open = False
            self._ensure_preview_window()
            return
        # Don't resize or move - let the window keep its current user-set geometry
        self._preview_visible = True

    def _ensure_preview_window(self):
        from neuroscience_sleep_scoring import SWS_utils
        if self.preview_window_open:
            return
        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
        # Size the window to the video's native frame size (not fullscreen).
        fw, fh = self._video_frame_size()
        if fw and fh:
            cv2.resizeWindow(self.preview_window_name, fw, fh)
        x = SWS_utils._video_window_props['x']
        y = SWS_utils._video_window_props['y']
        cv2.moveWindow(self.preview_window_name, x if x is not None else 0, y if y is not None else 0)
        try:
            cv2.setWindowProperty(self.preview_window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        # Add slider for video navigation
        self._video_slider_max = 1000
        self._video_max_time = 0
        if self.video_timestamp is not None and len(self.video_timestamp) > 0:
            self._video_max_time = self.video_timestamp['Offset_Time'].max()
        cv2.createTrackbar('Time', self.preview_window_name, 0, self._video_slider_max, self._on_video_slider)
        self.preview_window_open = True
        self._preview_visible = True

    def _update_preview_window_props(self):
        from neuroscience_sleep_scoring import SWS_utils
        try:
            rect = cv2.getWindowImageRect(self.preview_window_name)
            SWS_utils._video_window_props['x'] = rect[0]
            SWS_utils._video_window_props['y'] = rect[1]
            SWS_utils._video_window_props['width'] = rect[2]
            SWS_utils._video_window_props['height'] = rect[3]
        except Exception:
            pass




    def in_axes(self, event):

        # Add the crosshair here? TODO: put in priint statements to see when this triggers

        if DEBUG: print('scanning axes')

        #Stashing cursor thread here: https://stackoverflow.com/questions/63195460/how-to-have-a-fast-crosshair-mouse-cursor-for-subplots-in-matplotlib
        if event.inaxes == self.ax2:
            if DEBUG: print('Second bins')

        if event.inaxes == self.ax4:
            if DEBUG: print('EMG bin!!')

            #Should we call a graph refresh here?

            # x, y2 = sel.target
            # y1 = np.interp( sel.target[0],   plot1.get_xdata(), plot1.get_ydata() )
            # sel.annotation.set_text(f'x: {x:.2f}\ny1: {y1:.2f}\ny2: {y2:.2f}')
            # # sel.annotation.set_visible(False)
            # hline1 = ax1.axhline(y1, color='k', ls=':')
            # vline1 = ax1.axvline(x, color='k', ls=':')
            # vline2 = ax2.axvline(x, color='k', ls=':')
            # hline2 = ax2.axhline(y2, color='k', ls=':')
            # sel.extras.append(hline1)
            # sel.extras.append(vline1)
            # sel.extras.append(hline2)
            # sel.extras.append(vline2)
            #
            # fig = plt.figure(figsize=(15, 10))
            # ax1 = plt.subplot(2, 1, 1)
            # ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            #
            # plot1, = ax1.plot(np.array(np.random.uniform(-1, 1, 100).cumsum()))
            # plot2, = ax2.plot(np.array(np.random.uniform(-1, 1, 100).cumsum()))
            #
            # cursor = mplcursors.cursor(plot2, hover=True)
            # cursor.connect('add', crosshair)


        # Movie mode triggers when you hover over the bottom axis. Duh ax3 I guess
        # if event.inaxes == self.ax3:
        #     self.movie_mode = True
        #     print('MOVIE MODE!')
        # else:
        #     self.movie_mode = False


    def pull_up_movie(self, event):

        # I don't think we call movies there TODO: See if this was a stub for something else?
        if DEBUG: print('gon pull up some movies')

    ##
    def crosshair():


        plt.show()


    ##


    def on_click(self, event):

        if DEBUG: print("self.clicked = " + str(self.clicked))

        # Ignore clicks in fig2 axes
        if event.inaxes in self.fig2_axes:
            return

        # Microarousal placement: if armed via 'm', a click on the scoring plot
        # drops micro_size Wake bins starting at the clicked epoch, then disarms.
        if self.micro_mode:
            if event.inaxes == self.ax2 and event.xdata is not None:
                start = int(math.floor(event.xdata))
                if start < 0:
                    start = 0
                self.bins = [start, start + self.micro_size]
                self.forced_state = 1
                self.change_bins = True
                self.clicked = False
                self.micro_mode = False
                print(f'Microarousal placed: {self.micro_size} Wake bin(s) at {start}.')
            else:
                print('Click on the scoring plot (states panel) to place the microarousal.')
            return

        if self.movie_mode:
            self.movie_bin = event.xdata
            if DEBUG:
                print(f'video bin (xdata): {event.xdata}')
                print(f'x: {event.x}')
        elif self.clicked:
            if DEBUG: print('click registered')
            if event.inaxes == self.ax1:
                # Allow spectrogram clicks even if a bin selection is pending
                self.clicked = False
                self.bins = []
                self.replot = True
                self.replotx = event.xdata
                return
            if event.inaxes == self.ax2:
                if self.clicked == True:
                    if DEBUG: print(F'SECOND CLICK ----  xdata:{event.xdata} x:{event.x} axes: {event.inaxes}')
                    self.popup_xy = self._get_screen_xy(event)
                    self.bins.append(math.floor(event.xdata))
                    # Sort bins so start < end regardless of click order
                    self.bins = sorted(self.bins)
                    self.clicked = False
                    self.change_bins = True
        else:
            if event.inaxes == self.ax1:
                if DEBUG: print('Inside Spectrogram')

                #Set this bool to true, and then have it get flipped back to false in New_SWS
                self.replot = True

                if DEBUG:
                    print(f'event.x: {event.x}')
                    print(f'event.xdata {event.xdata}')

                #Set a replot start
                self.replotx = event.xdata

                #Log and store the xpos

                # Replot the graph here
            else:
                if DEBUG: print('Clicked outside of any bins')
            if event.inaxes != self.ax2:
                if DEBUG: print('please click in the second figure to select bins')
            else:
                self.bins.append(math.floor(event.xdata))
                if DEBUG: print(f'FIRST CLICK ----- xdata:{event.xdata} x:{event.x} axes: {event.inaxes}')
                self.popup_xy = self._get_screen_xy(event)
                self.clicked = True

    def _get_screen_xy(self, event):
        gui = getattr(event, 'guiEvent', None)
        if gui is None:
            return None
        if hasattr(gui, 'x_root') and hasattr(gui, 'y_root'):
            return (gui.x_root, gui.y_root)
        if hasattr(gui, 'globalX') and hasattr(gui, 'globalY'):
            return (gui.globalX(), gui.globalY())
        if hasattr(gui, 'globalPos'):
            pos = gui.globalPos()
            return (pos.x(), pos.y())
        return None

class ScoringCursor(object):
    def __init__(self, ax1, ax2, ax4):
        self.clicked=False
        self.second_click = False
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax4 = ax4
        self.movie_mode = False
        self.bins = []
        self.change_bins = False
        self.movie_bin = 0
        self.DONE = False
        self.STATE = []

        # Replot flags
        self.replot = False
        self.replotx = 0

        # Blitting support
        self.background = None

        self.horizontal_line = ax2.axhline(color='k', lw=0.8, ls='--', animated=True)
        self.vertical_line = ax2.axvline(color='k', lw=0.8, ls='--', animated=True)
        self.text = ax2.text(0.72, 0.9, '', transform=ax2.transAxes, animated=True)

        # initializing the lines
        self.ylims_ax1 = ax1.get_ylim()
        self.ylims_ax2 = ax2.get_ylim()

        line1 = ax1.plot([0,0], [self.ylims_ax1[0], self.ylims_ax1[1]], linewidth = 0.5, color = 'k')
        ml1 = line1.pop(0)

        line2 = ax2.plot([0,0], [self.ylims_ax2[0], self.ylims_ax2[1]], linewidth = 0.5, color = 'k')
        ml2 = line2.pop(0)

        self.movement_x_axis = np.linspace(0,60,900)
        self.spect_x_axis = np.linspace(199,1442, 900)

        self.lines = [ml1, ml2]
        self.toggle_line = False

        # Connect resize event to invalidate blitting cache:
        # fig.canvas.mpl_connect('resize_event', cursor.on_resize)

        if DEBUG: print('making a cursor')


    def on_resize(self, event):
        """Invalidate blitting background on window resize."""
        self.background = None

    def on_move(self, event):
        if DEBUG: print('on move')

    # def on_press(self, event):
    #     if event.key == 'd':
    #         print('DONE SCORING')
    #         self.DONE = True
    #     elif event.key in [1,2,3,4]:
    #         self.STATE.append(event.key)
    #     elif event.key == 'l':
    #         print(f'toggling line!! xdata: {event.xdata} ydata: {event.ydata}')
    #         for line in self.lines:
    #             line.remove()
    #         line1 = self.ax1.plot([self.spect_x_axis[int(event.xdata)],self.spect_x_axis[int(event.xdata)]], [self.ylims_ax1[0], self.ylims_ax1[1]], linewidth = 0.5, color = 'k')
    #         line2 = self.ax2.plot([int(event.xdata), int(event.xdata)], [self.ylims_ax2[0], self.ylims_ax2[1]], linewidth = 0.5, color = 'k')
    #         line3 = self.ax3.plot([self.movement_x_axis[int(event.xdata)],self.movement_x_axis[int(event.xdata)]], [self.ylims_ax3[0], self.ylims_ax3[1]], linewidth = 0.5, color = 'k')
    #         self.lines[0] = line1.pop(0)
    #         self.lines[1] = line2.pop(0)
    #         self.lines[2] = line3.pop(0)

    # def on_mouse_move(self, event):
    #     if not event.inaxes:
    #         need_redraw = self.set_cross_hair_visible(False)
    #         if need_redraw:
    #             self.ax2.figure.canvas.draw()
    #     else:
    #         self.set_cross_hair_visible(True)
    #         x, y = event.xdata, event.ydata
    #         # update the line positions
    #         self.horizontal_line.set_ydata(y)
    #         self.vertical_line.set_xdata(x)
    #         self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
    #         self.ax2.figure.canvas.draw()




    def in_axes(self, event):

        # Add the crosshair here? TODO: put in priint statements to see when this triggers

        if DEBUG: print('scanning axes')

        #Stashing cursor thread here: https://stackoverflow.com/questions/63195460/how-to-have-a-fast-crosshair-mouse-cursor-for-subplots-in-matplotlib

            #Should we call a graph refresh here?

            # x, y2 = sel.target
            # y1 = np.interp( sel.target[0],   plot1.get_xdata(), plot1.get_ydata() )
            # sel.annotation.set_text(f'x: {x:.2f}\ny1: {y1:.2f}\ny2: {y2:.2f}')
            # # sel.annotation.set_visible(False)
            # hline1 = ax1.axhline(y1, color='k', ls=':')
            # vline1 = ax1.axvline(x, color='k', ls=':')
            # vline2 = ax2.axvline(x, color='k', ls=':')
            # hline2 = ax2.axhline(y2, color='k', ls=':')
            # sel.extras.append(hline1)
            # sel.extras.append(vline1)
            # sel.extras.append(hline2)
            # sel.extras.append(vline2)
            #
            # fig = plt.figure(figsize=(15, 10))
            # ax1 = plt.subplot(2, 1, 1)
            # ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            #
            # plot1, = ax1.plot(np.array(np.random.uniform(-1, 1, 100).cumsum()))
            # plot2, = ax2.plot(np.array(np.random.uniform(-1, 1, 100).cumsum()))
            #
            # cursor = mplcursors.cursor(plot2, hover=True)
            # cursor.connect('add', crosshair)


        # Movie mode triggers when you hover over the bottom axis. Duh ax3 I guess
        # if event.inaxes == self.ax3:
        #     self.movie_mode = True
        #     print('MOVIE MODE!')
        # else:
        #     self.movie_mode = False


    # def pull_up_movie(self, event):

    #     # I don't think we call movies there TODO: See if this was a stub for something else?
    #     print('gon pull up some movies')

    # ##
    # def crosshair():


    #     plt.show()


    # ##


    def on_click(self, event):

        if DEBUG: print("self.clicked = " + str(self.clicked))

        if event.inaxes == self.ax1:
            if DEBUG: print('Inside Spectrogram')

            #Set this bool to true, and then have it get flipped back to false in New_SWS
            self.replot = True

            if DEBUG:
                print(f'event.x: {event.x}')
                print(f'event.xdata {event.xdata}')

            #Set a replot start
            self.replotx = event.xdata
