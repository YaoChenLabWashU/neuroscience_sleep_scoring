import ChenLab_Sleep_Scoring as slp
filename_sw = '/scratch/ltilden/ChenLab_Sleep_Scoring/KQ2.json'
slp.downsample_filter(filename_sw)
slp.create_spectrogram(filename_sw)
