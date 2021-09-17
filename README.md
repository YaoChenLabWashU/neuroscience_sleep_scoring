Sleep Scoring 1.4.0 Latest Release:
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Sleep Scoring Workflow:
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
To start a new project:
1) Adjust your Sleep_Scoring.json file (I recommend making a local copy for yourself on your desktop)
- Below is an example of the .json file with explanations of each variable

{
  "rawdat_dir": "/Volumes/yaochen/Active/Jacob Amme/jaLC_FLiPAKAREEGEMG004/jaLC_FLiPAKAREEGEMG004_data/", #This is where your raw EEG is
  "model_dir": "/Users/lizzie/Desktop/Remote_Git/neuroscience_sleep_scoring/model/", #This is the directory with the Sleep Scoring model
  "video_dir": "/Users/lizzie/Box/ChenLab/Jacob Amme/FLiP Videos/jaLC_FLiPAKAREEGEMG004/", #This is the directory wiht videos
  "log_dir": "/Users/lizzie/Desktop/", #This is where the log file is saved
  "animal": "1225-9", #Animal number
  "mod_name": "mouse", #I wouldn't change this
  "mouse_name": "1225-9", #Animal name (not experiment name)
  "epochlen": 4, #This is the bin size for scoring (cannot go lower than 4)
  "fs": 400, #Sampling rate of raw data
  "fsd": 200, #Downsampled rate
  "emg": 1, #1 if you used EMG, 0 if you did not
  "vid": 1, #if you used motion tracking, 0 if you did not
  "movement": 1, #if you used video, 0 if you did not
  "EEG channel": 0, #Auxillary channel that will be used for the spectrogram (either 0 or 2)
  "EMG channel": 3, #Auxillary channel that will be used for the EMG
  "Acquisition": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24
  ], #Which acquisitions you would like to score
  "Filter High": 100, #High-pass filter for the EEG and EMG 
  "Filter Low": 0.5, #Low-pass filter for the EEG and EMG 
  "savedir": "/Volumes/yaochen/Active/Jacob Amme/jaLC_FLiPAKAREEGEMG004/jaLC_FLiPAKAREEGEMG004_extracted_data/", #directory where your extracted data will go
  "Bonsai Version": 3 #The Bonsai workflow that you used for the experiment
}

2)Run the command below to extract and downsample the lfp

python /path/to/sleep/scoring/package/extract_data.py /path/to/json/file

3) Once you have run the above line (only once per experiment), run the lone below to start the sleep scoring engine:

python /Users/lizzie/Desktop/Remote_Git/neuroscience_sleep_scoring/New_SWS.py /Users/lizzie/Desktop/Remote_Git/neuroscience_sleep_scoring/Score_Setting.json

4) Follow the prompts the load up the acquisition that you want

A few notes about the program:
- Scoring Mode should be used for a new hour
- Checking Mode should be used for an hour that has no scoring
- Press "d" to exit checking mode and "q" to exit scoring mode
- Press "v" to click out of a movie early



