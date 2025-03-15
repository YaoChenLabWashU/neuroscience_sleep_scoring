import time
import os
import numpy as np
import glob
import pandas as pd
import scipy.io

these_experiments = glob.glob('/Volumes/yaochen/Active/Lizzie/FLP_data/ltFLiPAKAREEGEMG00*')
basenames = [these_experiments[i].split('/')[-1] for i in range(len(these_experiments))]
time_dict = {}
for e in np.arange(0, len(these_experiments)):
	this_dict = {}
	this_dict['Timestamp Times'] = []
	this_dict['AD Times'] = []

	video_dir  = os.path.join('/Volumes/ChenLabHDDC/FLiP_Videos/', basenames[e], basenames[e] + '_video')
	csv_dir = os.path.join('/Volumes/ChenLabHDDC/FLiP_Videos/', basenames[e], basenames[e] + '_csv')
	timestamp_files = glob.glob(os.path.join(csv_dir, '*timestamp*.csv'))
	timestamp_files.sort(key=lambda f: os.path.getmtime(os.path.join(csv_dir, f)))
	for t in timestamp_files:
		print('Timestamp file: ' + t)
		timestamp_df = pd.read_csv(t, delimiter = "\n", header=None)
		timestamp_df.columns = ['Timestamps']
		last_frame = timestamp_df['Timestamps'].iloc[-1]
		ts_format = '%Y-%m-%dT%H:%M:%S'
		short_ts = last_frame[:last_frame.find('.')]
		datetime_timestamp = datetime.strptime(short_ts, ts_format)
		this_dict['Timestamp Times'].append(datetime_timestamp)

	AD_files = glob.glob(os.path.join(these_experiments[e], 'AD0_*.mat'))
	try:
		AD_files.remove('/Volumes/yaochen/Active/Lizzie/FLP_data/'+basenames[e]+'/AD0_e1p6avg.mat')
	except ValueError:
		print('skipping these experiment')
		continue
	AD_files.sort(key=lambda f: os.path.getmtime(os.path.join(these_experiments[e], f)))

	for a in AD_files:
		if os.path.getsize(a) < 5000000:
			continue
		print('AD file: ' + a)
		mod_time = time.ctime(os.path.getmtime(a))
		ts_format = '%a %b %d %H:%M:%S %Y'
		datetime_AD = datetime.strptime(mod_time, ts_format)
		this_dict['AD Times'].append(datetime_AD)
	time_dict[basenames[e]] = this_dict
	autonotes = scipy.io.loadmat(os.path.join(these_experiments[e], 'autonotes.mat'))['notebook'][0]
	mat_ts = [autonotes[i][0][:8] for i in range(np.size(autonotes))]
	ts_format = '%H:%M:%S'
	datetime_mat = [datetime.strptime(mat_ts[i], ts_format) for i in range(np.size(autonotes))]
	time_dict[basenames[e]]['Mat Times'] = datetime_mat
for exp in list(time_dict.keys()):
	time_dict[exp]['Mat-AD Difference'] = []
	time_dict[exp]['TS-AD Difference'] = []
	time_dict[exp]['TS-Mat Difference'] = []
	for i in range(len(time_dict[exp]['AD Times'])):
		this_AD_time = time_dict[exp]['AD Times'][i]
		this_mat_time = time_dict[exp]['Mat Times'][i]
		this_ts_time = time_dict[exp]['Timestamp Times'][i]
		this_mat_time = datetime.combine(this_AD_time.date(), this_mat_time.time())
		time_dict[exp]['Mat-AD Difference'].append((this_mat_time-this_AD_time).total_seconds())
		time_dict[exp]['TS-AD Difference'].append((this_ts_time-this_AD_time).total_seconds())
		time_dict[exp]['TS-Mat Difference'].append((this_ts_time-this_mat_time).total_seconds())







