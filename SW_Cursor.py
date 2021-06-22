import math
import matplotlib.pyplot as plt
import numpy as np


#Cursor class for scoring figure 2 in the sleeps corind with the spectrogram and predicted stats

class Cursor(object):
    def __init__(self, ax1, ax2, ax3):
        self.clicked=False
        self.second_click = False
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.movie_mode = False
        self.bins = []
        self.change_bins = False
        self.movie_bin = 0
        self.DONE = False
        self.STATE = []

        self.horizontal_line = ax2.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax2.axvline(color='k', lw=0.8, ls='--')
        self.text = ax2.text(0.72, 0.9, '', transform=ax2.transAxes)

        # initializing the lines
        self.ylims_ax1 = ax1.get_ylim()
        self.ylims_ax2 = ax2.get_ylim()
        self.ylims_ax3 = ax3.get_ylim()

        line1 = ax1.plot([0,0], [self.ylims_ax1[0], self.ylims_ax1[1]], linewidth = 0.5, color = 'k')
        ml1 = line1.pop(0)

        line2 = ax2.plot([0,0], [self.ylims_ax2[0], self.ylims_ax2[1]], linewidth = 0.5, color = 'k')
        ml2 = line2.pop(0)

        line3 = ax3.plot([0,0], [self.ylims_ax3[0], self.ylims_ax3[1]], linewidth = 0.5, color = 'k')
        ml3 = line3.pop(0)

        self.movement_x_axis = np.linspace(0,60,900)
        self.spect_x_axis = np.linspace(199,1442, 900)

        self.lines = [ml1, ml2, ml3]
        self.toggle_line = False

        print('making a cursor')


    def on_move(self, event):
        print('on move')

    def on_press(self, event):
        if event.key == 'd':
            print('DONE SCORING')
            self.DONE = True
        elif event.key in [1,2,3,4]:
            self.STATE.append(event.key)
        elif event.key == 'l':
            print(f'toggling line!! xdata: {event.xdata} ydata: {event.ydata}')
            for line in self.lines:
                line.remove()
            line1 = self.ax1.plot([self.spect_x_axis[int(event.xdata)],self.spect_x_axis[int(event.xdata)]], [self.ylims_ax1[0], self.ylims_ax1[1]], linewidth = 0.5, color = 'k')
            line2 = self.ax2.plot([int(event.xdata), int(event.xdata)], [self.ylims_ax2[0], self.ylims_ax2[1]], linewidth = 0.5, color = 'k')
            line3 = self.ax3.plot([self.movement_x_axis[int(event.xdata)],self.movement_x_axis[int(event.xdata)]], [self.ylims_ax3[0], self.ylims_ax3[1]], linewidth = 0.5, color = 'k')
            self.lines[0] = line1.pop(0)
            self.lines[1] = line2.pop(0)
            self.lines[2] = line3.pop(0)


    # This works, but doesn't refresh fast enough. I think this is a limit of matplotlib however and out of my control
    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax2.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax2.figure.canvas.draw()




    def in_axes(self, event):

        # Add the crosshair here? TODO: put in priint statements to see when this triggers



        #Stashing cursor thread here: https://stackoverflow.com/questions/63195460/how-to-have-a-fast-crosshair-mouse-cursor-for-subplots-in-matplotlib
        #this would be so much chooler if I could use a switch statement but fuck it
        if event.inaxes == self.ax2:

            print('Second bins')
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
        if event.inaxes == self.ax3:
            self.movie_mode = True
            print('MOVIE MODE!')
        else:
            self.movie_mode = False


    def pull_up_movie(self, event):

        # I don't think we call movies there TODO: See if this was a stub for something else?
        print('gon pull up some movies')

    ##
    def crosshair():


        plt.show()


    ##


    def on_click(self, event):
        if self.movie_mode:
            self.movie_bin = event.xdata
            print(f'video bin (xdata): {event.xdata}')
            print(f'x: {event.x}')
        elif self.clicked:
            if event.inaxes != self.ax2:
                print('please click in the second figure to select bins')
            else:
                print(F'SECOND CLICK ----  xdata:{event.xdata} x:{event.x} axes: {event.inaxes}')
                self.bins.append(math.floor(event.xdata))
                self.clicked = False
                self.change_bins = True
        else:
            if event.inaxes != self.ax2:
                print('please click in the second figure to select bins')
            else:
                self.bins.append(math.floor(event.xdata))
                print(f'FIRST CLICK ----- xdata:{event.xdata} x:{event.x} axes: {event.inaxes}')
                self.clicked = True
