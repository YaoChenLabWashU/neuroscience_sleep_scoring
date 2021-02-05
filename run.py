import os
os.chdir('/Users/annzhou/research/neuroscience/')
#import ChenLab_Sleep_Scoring as slp
import New_SWS
filename_sw = '/Users/annzhou/research/neuroscience/ChenLab_Sleep_Scoring/Score_Settings.json'
#slp.load_data_for_sw(filename_sw)
New_SWS.load_data_for_sw(filename_sw)