import math
import matplotlib.pyplot as plt
import numpy as np

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

    def in_axes(self, event):
        if event.inaxes == self.ax3:
            self.movie_mode = True
            print('MOVIE MODE!')
        else:
            self.movie_mode = False
    def pull_up_movie(self, event):
        print('gon pull up some movies')


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




