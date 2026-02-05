import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

DEBUG = False

class Cursor(object):
    def __init__(self, ax1, ax2, ax4, all_axes=None, epochlen=4):
        self.clicked=False
        self.second_click = False
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax4 = ax4
        self.all_axes = all_axes if all_axes else [ax1, ax2, ax4]
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

        # Preview mode (p key) - drag to see video frames
        self.preview_mode = False
        self.video_cap = None
        self.video_timestamp = None
        self.video_d = None
        self.preview_window_name = 'Preview'
        self.preview_window_open = False
        self.last_preview_frame_idx = None
        
        # Magnify mode (m key) - show zoomed view around cursor
        self.magnify_mode = False
        self.magnify_callback = None  # Callback to update magnified view

        # Blitting support
        self.background = None

        # Create vertical lines for ALL axes (for full-height crosshair)
        self.vertical_lines = []
        for ax in self.all_axes:
            vline = ax.axvline(color='k', lw=0.8, ls='--', animated=True)
            self.vertical_lines.append(vline)
        
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

        self.movement_x_axis = np.linspace(0,60,900)
        self.spect_x_axis = np.linspace(199,1442, 900)

        self.lines = [ml1, ml2]
        self.toggle_line = False

        if DEBUG: print('making a cursor')


    def on_resize(self, event):
        """Invalidate blitting background on window resize."""
        self.background = None

    def on_move(self, event):
        if DEBUG: print('on move')

    def on_press(self, event):
        if event.key == 'd':
            if DEBUG: print('DONE SCORING')
            self.DONE = True
        elif event.key == 'p':
            self.preview_mode = not self.preview_mode
            if self.preview_mode:
                print('Preview mode ON - move on spectrogram to preview')
                self._ensure_preview_window()
            else:
                print('Preview mode OFF')
                if self.preview_window_open:
                    try:
                        cv2.destroyWindow(self.preview_window_name)
                    except Exception:
                        pass
                    self.preview_window_open = False
            self.background = None  # Invalidate to recapture
        elif event.key == 'm':
            # Toggle magnify mode
            self.magnify_mode = not self.magnify_mode
            if self.magnify_mode:
                print('Magnify mode ON - drag cursor to see ±30s zoomed view')
            else:
                print('Magnify mode OFF')
            self.background = None
        elif event.key in [1,2,3,4]:
            self.STATE.append(event.key)
        elif event.key == 'l':
            if DEBUG: print(f'toggling line!! xdata: {event.xdata} ydata: {event.ydata}')
            for line in self.lines:
                line.remove()
            line1 = self.ax1.plot([self.spect_x_axis[int(event.xdata)],self.spect_x_axis[int(event.xdata)]], [self.ylims_ax1[0], self.ylims_ax1[1]], linewidth = 0.5, color = 'k')
            line2 = self.ax2.plot([int(event.xdata), int(event.xdata)], [self.ylims_ax2[0], self.ylims_ax2[1]], linewidth = 0.5, color = 'k')
            self.lines[0] = line1.pop(0)
            self.lines[1] = line2.pop(0)


    def _x_for_axis(self, ax, x):
        if ax == self.ax2:
            return x / self.epochlen
        return x

    def _event_x_to_seconds(self, event):
        if event.inaxes == self.ax2:
            return event.xdata * self.epochlen
        return event.xdata

    def _set_cursor_x(self, x, y=None):
        if y is not None:
            self.horizontal_line.set_ydata([y, y])
        for i, vline in enumerate(self.vertical_lines):
            ax = self.all_axes[i]
            x_mapped = self._x_for_axis(ax, x)
            vline.set_xdata([x_mapped, x_mapped])

    # This works, but doesn't refresh fast enough. I think this is a limit of matplotlib however and out of my control
    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        for vline in self.vertical_lines:
            vline.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        # Initialize background for blitting on first call
        if self.background is None:
            self.ax2.figure.canvas.draw()
            self.background = self.ax2.figure.canvas.copy_from_bbox(self.ax2.figure.bbox)

        if not event.inaxes:
            if self.horizontal_line.get_visible():
                self.set_cross_hair_visible(False)
                self.ax2.figure.canvas.restore_region(self.background)
                self.ax2.figure.canvas.blit(self.ax2.figure.bbox)
        else:
            if not self.horizontal_line.get_visible():
                self.set_cross_hair_visible(True)
            x_sec = self._event_x_to_seconds(event)
            y = event.ydata
            
            # Update the line positions - vertical line across all axes
            self._set_cursor_x(x_sec, y=y)
            self.text.set_text('x=%1.2f, y=%1.2f' % (x_sec, y))

            self.ax2.figure.canvas.restore_region(self.background)
            self.ax2.draw_artist(self.horizontal_line)
            for i, vline in enumerate(self.vertical_lines):
                self.all_axes[i].draw_artist(vline)
            self.ax2.draw_artist(self.text)
            self.ax2.figure.canvas.blit(self.ax2.figure.bbox)
            
            # Preview mode: show video frame at cursor position
            if self.preview_mode and event.inaxes == self.ax1 and self.video_cap is not None:
                try:
                    self._show_preview_frame(x_sec)
                except Exception:
                    self.preview_mode = False
                    self.background = None

            # Magnify mode: update zoomed view
            if self.magnify_mode and event.inaxes == self.ax1 and self.magnify_callback is not None:
                self.magnify_callback(x_sec)

    def _show_preview_frame(self, time_sec):
        """Show video frame at the given time in seconds."""
        if self.video_timestamp is None or self.video_cap is None:
            return
        try:
            # Find frame index for this time
            idx = self.video_timestamp.index[self.video_timestamp['Offset_Time'] <= time_sec]
            if len(idx) == 0:
                return
            frame_idx = idx[-1]

            if self.last_preview_frame_idx == frame_idx:
                return
            self.last_preview_frame_idx = frame_idx
            
            # Get video filename and read frame
            from neuroscience_sleep_scoring import SWS_utils
            v = SWS_utils.get_videofn_from_csv(self.video_d, self.video_timestamp['Filename'][frame_idx])
            if v in self.video_cap and self.video_cap[v].isOpened():
                self.video_cap[v].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.video_cap[v].read()
                if ret:
                    self._ensure_preview_window()
                    bin_idx = int(time_sec // self.epochlen)
                    label = f"t={time_sec:.2f}s  bin={bin_idx}  frame={frame_idx}"
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow(self.preview_window_name, frame)
                    cv2.waitKey(1)
                    self._update_preview_window_props()
        except Exception as e:
            if DEBUG: print(f'Preview error: {e}')

    def _ensure_preview_window(self):
        from neuroscience_sleep_scoring import SWS_utils
        if self.preview_window_open:
            return
        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
        if SWS_utils._video_window_props['width'] is not None and SWS_utils._video_window_props['height'] is not None:
            cv2.resizeWindow(self.preview_window_name, SWS_utils._video_window_props['width'], SWS_utils._video_window_props['height'])
        else:
            cv2.resizeWindow(self.preview_window_name, 640, 480)
        if SWS_utils._video_window_props['x'] is not None and SWS_utils._video_window_props['y'] is not None:
            cv2.moveWindow(self.preview_window_name, SWS_utils._video_window_props['x'], SWS_utils._video_window_props['y'])
        try:
            cv2.setWindowProperty(self.preview_window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        self.preview_window_open = True

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

        if self.preview_mode and event.inaxes == self.ax1:
            if event.xdata is None:
                return
            time_sec = self._event_x_to_seconds(event)
            self._show_preview_frame(time_sec)
            self._set_cursor_x(time_sec)
            self.preview_mode = False
            self.background = None
            if DEBUG: print('Preview mode OFF')
            return

        if self.movie_mode:
            self.movie_bin = event.xdata
            if DEBUG:
                print(f'video bin (xdata): {event.xdata}')
                print(f'x: {event.x}')
        elif self.clicked:
            if DEBUG: print('click registered')
            if event.inaxes == self.ax2:
                if self.clicked == True:
                    if DEBUG: print(F'SECOND CLICK ----  xdata:{event.xdata} x:{event.x} axes: {event.inaxes}')
                    self.popup_xy = self._get_screen_xy(event)
                    self.bins.append(math.floor(event.xdata))
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
