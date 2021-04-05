

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - ###
### For firsttime users:

## Make sure that you have a CHPC account with anaconda3 installed 

## copy the Chen Lab Scoring folder from the server to your chpc account:

scp -r /Volumes/yaochen/Active/CENTRAL_CODE/ChenLab_Sleep_Scoring/ username@login01.chpc.wustl.edu:/scratch/username/

## logon to the CHPC

ssh username@login01.chpc.wustl.edu

# Copy the extract_lfp.sh and extract_lfp.py to your home directory 

cp /scratch/username/ChenLab_Sleep_Scoring/extract_lfp.* /home/username/

# Edit the extract_lfp.sh and extract_plf.py

vim /home/username/extract_lfp.sh

Example:
#!/bin/bash


#PBS -N Extracting 

#PBS -l nodes=2:ppn=4:gpus=1,walltime=03:00:00

#module load cuda-9.0p1
#module load cuDNN-7.1.1

. /scratch/khengen_lab/anaconda3/etc/profile.d/conda.sh (**Edit this to your anaconda path**)

python /home/ltilden/extract_lfp.py &> /home/ltilden/extractlfp.log (**Edit this to your own home directory**)

# Edit your .bashrc file to include ChenLab_Sleep_Scoring in python path
vim ~/.bashrc
Example:

# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# User specific aliases and functions
source /act/etc/profile.d/actbin.sh

export PYTHONPATH="${PYTHONPATH}:/scratch/ltilden" 
# added by Anaconda3 2018.12 installer
# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '/scratch/khengen_lab/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/scratch/khengen_lab/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/scratch/khengen_lab/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/scratch/khengen_lab/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - ###

### For every time you are scoring a new animal

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -##
### Step 1: Extract data from matlab files
## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -##

## logon to the CHPC:

ssh username@login01.chpc.wustl.edu

# Create a directory that will hold your raw data

mkdir /scratch/username/raw_data_folder/

## Exit the CHPC

exit

# copy your raw data onto the CHPC:

scp /Path/To/MATLAB/data/AD*.mat username@login01.chpc.wustl.edu:/scratch/username/raw_data_folder/

# Edit the json script (/Volumes/yaochen/Active/CENTRAL_CODE/ChenLab_Sleep_Scoring/Score_Settings.json)

Example:
{"rawdat_dir" : "/Volumes/yaochen/Active/Yao/EEGEMG/ycEEGEMG001/",
 "model_dir" : "/Volumes/yaochen/Active/CENTRAL_CODE/",
 "animal": "ycEEGEMG001",
 "epochlen" : 4,
 "fs" : 400,
 "fsd": 200,
 "emg" : 1 ,
 "vid": 0,
 "EEG channel": 0,
 "EMG channel": 3,
 "Acquisition": [],
 "Filter High": 100,
 "Filter Low": 0.5,
 "savedir": "/Volumes/yaochen/Active/Yao/EEGEMG/ycEEGEMG001/extracted_data/"
}

# choose your acquisition

ipython


import os
os.chdir('/Volumes/yaochen/Active/CENTRAL_CODE')
import ChenLab_Sleep_Scoring as slp
filename_sw = '/Volumes/yaochen/Active/CENTRAL_CODE/ChenLab_Sleep_Scoring/Score_Settings.json'
slp.choosing_acquisition(filename_sw)

#copy the resulting array into the .json scripts

## logon to the CHPC:

ssh username@login01.chpc.wustl.edu

# Edit the json script (this has all the settings for the scoring job)

vim /scratch/username/ChenLab_Sleep_Scoring/Score_Settings.json

Example:
{"rawdat_dir" : "/scratch/ltilden/FLiP_data/ycEEGEMG001/",
 "model_dir" : "/scratch/ltilden/ChenLab_Sleep_Scoring/", (#This isn't important yet)
 "animal": "ycEEGEMG001/",
 "epochlen" : 4,
 "fs" : 400,
 "fsd": 200,
 "emg" : 1 ,
 "vid": 0,
 "EEG channel": 0,
 "EMG channel": 3,
 "Acquisition": [2, 5, 6, 8],
 "Filter High": 100,
 "Filter Low": 0.5,
 "savedir": "/scratch/ltilden/FLiP_data/ycEEGEMG001/extracted_data/"
}

# Edit extraction job

vim /home/username/extract_lfp.py
change "filename_sw = '/scratch/username/ChenLab_Sleep_Scoring/Score_Settings.json'"

# bug fix

comment out "from .New_SWS import *" from __init__.py

# Submit your extraction job

qsub extract_lfp.sh
**You can check the status of your job by using: qstat -u username

Q-in queue
R-still running
C-complete

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -##
### Step 2: Copy extracted files back to the server
## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -##

## Exit the CHPC
exit
## Copy extracted files:
scp -r ltilden@login01.chpc.wustl.edu:/CHPC/path/to/raw/data/extracted_data/ /Computer/path/to/raw/data/ycEEGEMG001/

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -##
### Step 3: Make sure your videos are in an mp4 format
## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -##

# If the files are in .avi, you will have to convert them

# To convert all of the files in your video directory:
# move to your video directory
cd /path/to/your/video
# copy and paste below
for i in *.avi; do ffmpeg -i "$i" "${i%.*}.mp4"; done
# To convert only a specific file:
ffmpeg -i yourfile.avi newfilename.mp4


#### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#### Step 4: Set up the file directory
#### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

The basic principle here is that your `Score_Settings.json` must correspond to your directory structure. In addition, take note of the path to your `Score_Settings.json`; we will use it later.

An example below:

Directory:<br>
home/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ChenLab_Sleep_Scoring<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Score_Settings.json <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model/ (all model stuff goes here) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;initial_data/ (place all initial data for training the first model under this directory) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(You should put the model here. / You can find the model here.) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;jaLC_FLiPAKAREEGEMG004/ (all your data goes here)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;jaLC_FLiPAKAREEGEMG004_data/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;extracted_data/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(The result, for instance StateAcq1_hr0.npy, can be found here.) <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;jaLC_FLiPAKAREEGEMG004_0.mp4 (all movie files are listed here (not in any subdirectory)) <br>

Score_Settings.json:<br>
"rawdat_dir" : "/home/jaLC_FLiPAKAREEGEMG004/jaLC_FLiPAKAREEGEMG004_data/"<br>
 "model_dir" : "/home/ChenLab_Sleep_Scoring/model/"<br>
 "video_dir" : "/home/jaLC_FLiPAKAREEGEMG004/"<br>
 "savedir": "/home/jaLC_FLiPAKAREEGEMG004/jaLC_FLiPAKAREEGEMG004_data/extracted_data/"<br>
 
#### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#### Step 5: Train the first model
#### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

If you do not plan to use the model or you already have a model trained, skip to step 6.

Before training the first model, go to `train_model.py` and update `raw_data` under `train_first_model` function according to what you have. <br>
For instance, if you have scored states from acquisition1hour0, acquisition2hour0 and acquisition4hour0, your `raw_data` should look like:<br>
> raw_data = {<br>
> &nbsp;&nbsp;&nbsp;&nbsp;1:0,<br>
> &nbsp;&nbsp;&nbsp;&nbsp;2:0,<br>
> &nbsp;&nbsp;&nbsp;&nbsp;4:0<br>
> }<br>

Place your datasets (for instance, `StatesAcq1_hr0.npy`, `downsampEMG_Acq1_hr0.npy` and `downsampEEG_Acq1_hr0.npy` for each scoring) inside the initial_data folder. <br>
Go back to the home directory and run `python train_model.py`.

#### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#### Step 6: Score!
#### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

Run `python New_SWS.py [path to your Score_Settings.json]`. For instance, using the sample file directory above, you should run `python New_SWS.py /home/ChenLab_Sleep_Scoring/Score_Settings.json`.

When in the manual scoring mode, you can exit by typing `q`.
When in the fixing mode, you can exit by typing`d`.
(Note that in both cases, you need to be focused on the scoring window before typing `q` or `d`).

##Scoring
# 1 = Awake (Green)
# 2 = NREM (Blue)
# 3 = REM (Red)

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -##
### Notes
## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -##

# You can close the window at any time! It will automatically load the states that you have already scored the next time that you open!

# For the first and last state, make sure you click into the terminal window, type the state, and then press enter. All other states, make sure you are clicked into the spectrogram window, and just press the scoring numbers without pressing enter

# Debug log

## ModuleNotFoundError: No module named 'cv2'

`cd ChenLab_Sleep_Scoring`

`pip install opencv-python`



