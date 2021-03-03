import numpy as np
import scipy.signal as signal
import os
import math
import pandas as pd
import json

os.chdir('/Users/annzhou/research/neuroscience/jaLC_FLiPAKAREEGEMG004/model/initial_data')
import SWS_utils


def train_first_model(extracted_dir, epochlen, fsd, emg_flag, animal, model_dir, mod_name):

    # Data available for training for the first time ("#acquisition:#hr")
    raw_data = {
        1:0,
        2:0,
        4:0
    }

    # Using EMG data by default. (No video for now)
    final_features = ['Animal_Name', 'animal_num', 'State', 'delta_pre', 'delta_pre2',
                      'delta_pre3', 'delta_post', 'delta_post2', 'delta_post3', 'EEGdelta', 'theta_pre',
                      'theta_pre2', 'theta_pre3',
                      'theta_post', 'theta_post2', 'theta_post3', 'EEGtheta', 'EEGalpha', 'EEGbeta',
                      'EEGgamma', 'EEGnarrow', 'nb_pre', 'delta/theta', 'EEGfire', 'EEGamp', 'EEGmax',
                      'EEGmean', 'EMG']

    data = np.empty((len(final_features),0))

    for a in raw_data:
        h = raw_data[a]

        print('Handling acquisition '+str(a)+' hour '+str(h))
        print('Loading EEG and EMG....')
        downsampEEG = np.load(os.path.join(extracted_dir, 'downsampEEG_Acq' + str(a) + '.npy'))
        if emg_flag:
            downsampEMG = np.load(os.path.join(extracted_dir, 'downsampEMG_Acq' + str(a) + '.npy'))
        acq_len = np.size(downsampEEG) / fsd  # fs: sampling rate, fsd: downsampled sampling rate
        hour_segs = math.ceil(acq_len / 3600)  # acq_len in seconds, convert to hours
        print('This acquisition has ' + str(hour_segs) + ' segments.')

        for h in np.arange(hour_segs):
            this_eeg = np.load(os.path.join(extracted_dir, 'downsampEEG_Acq' + str(a) + '_hr' + str(h) + '.npy'))
            if emg_flag:
                this_emg = np.load(os.path.join(extracted_dir, 'downsampEMG_Acq' + str(a) + '_hr' + str(h) + '.npy'))
            # chop off the remainder that does not fit into the 4s epoch
            seg_len = np.size(this_eeg) / fsd
            nearest_epoch = math.floor(seg_len / epochlen)
            new_length = int(nearest_epoch * epochlen * fsd)
            this_eeg = this_eeg[0:new_length]

            os.chdir(extracted_dir)
            if emg_flag:
                EMGamp, EMGmax, EMGmean = SWS_utils.generate_signal(this_emg, epochlen, fsd)
            else:
                EMGamp = False

            EEGamp, EEGmax, EEGmean = SWS_utils.generate_signal(this_eeg, epochlen, fsd)

            EEGdelta, idx_delta = SWS_utils.bandPower(0.5, 4, this_eeg, epochlen, fsd)

            EEGtheta, idx_theta = SWS_utils.bandPower(4, 8, this_eeg, epochlen, fsd)

            EEGalpha, idx_alpha = SWS_utils.bandPower(8, 12, this_eeg, epochlen, fsd)

            EEGbeta, idx_beta = SWS_utils.bandPower(12, 30, this_eeg, epochlen, fsd)

            EEGgamma, idx_gamma = SWS_utils.bandPower(30, 80, this_eeg, epochlen, fsd)

            EEG_broadtheta, idx_broadtheta = SWS_utils.bandPower(2, 16, this_eeg, epochlen, fsd)

            EEGfire, idx_fire = SWS_utils.bandPower(4, 20, this_eeg, epochlen, fsd)

            EEGnb = EEGtheta / EEG_broadtheta  # narrow-band theta
            delt_thet = EEGdelta / EEGtheta  # ratio; esp. important

            EEGdelta = SWS_utils.normalize(EEGdelta)
            EEGalpha = SWS_utils.normalize(EEGalpha)
            EEGbeta = SWS_utils.normalize(EEGbeta)
            EEGgamma = SWS_utils.normalize(EEGbeta)
            EEGnb = SWS_utils.normalize(EEGnb)
            EEGtheta = SWS_utils.normalize(EEGtheta)
            EEGfire = SWS_utils.normalize(EEGfire)
            delt_thet = SWS_utils.normalize(delt_thet)

            # frame shifting
            delta_post, delta_pre = SWS_utils.post_pre(EEGdelta, EEGdelta)
            theta_post, theta_pre = SWS_utils.post_pre(EEGtheta, EEGtheta)
            delta_post2, delta_pre2 = SWS_utils.post_pre(delta_post, delta_pre)
            theta_post2, theta_pre2 = SWS_utils.post_pre(theta_post, theta_pre)
            delta_post3, delta_pre3 = SWS_utils.post_pre(delta_post2, delta_pre2)
            theta_post3, theta_pre3 = SWS_utils.post_pre(theta_post2, theta_pre2)
            nb_post, nb_pre = SWS_utils.post_pre(EEGnb, EEGnb)

            animal_name = np.full(np.size(delta_pre), animal)
            # Note: The second parameter depends on the actual animal name. For example, if the animal is "KNR00004", we
            # should use "animal[3:]" for "00004"; if the animal is "jaLC_FLiPAKAREEGEMG004", we should use "animal[19:]"
            # for "004".
            animal_num = np.full(np.shape(animal_name), int(animal[19:]))

            # os.chdir(model_dir)
            # if emg_flag:
            #     clf = load(mod_name + '_EMG.joblib')
            # else:
            #     clf = load(mod_name + '_no_EMG.joblib')

            # feature list
            FeatureList = []
            nans = np.full(np.shape(animal_name), np.nan)
            if emg_flag:
                FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3, delta_post, delta_post2, delta_post3,
                               EEGdelta,
                               theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2, theta_post3,
                               EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                               EEGmean, EMGamp]
            else:
                FeatureList = [animal_num, delta_pre, delta_pre2, delta_pre3, delta_post, delta_post2, delta_post3,
                               EEGdelta,
                               theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2, theta_post3,
                               EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                               EEGmean, nans]

            FeatureList_smoothed = []
            for f in FeatureList:
                FeatureList_smoothed.append(signal.medfilt(f, 5))
            Features = np.column_stack((FeatureList_smoothed))
            Features = np.nan_to_num(Features)

            inital_data_dir = model_dir+"initial_data/"
            State = np.load(os.path.join(inital_data_dir, 'StatesAcq' + str(a) + '_hr' + str(h) + '.npy'))

            data_addition = np.vstack(
                [animal_name, animal_num, State, delta_pre, delta_pre2, delta_pre3, delta_post,
                 delta_post2, delta_post3, EEGdelta, theta_pre, theta_pre2, theta_pre3, theta_post, theta_post2,
                 theta_post3,
                 EEGtheta, EEGalpha, EEGbeta, EEGgamma, EEGnb, nb_pre, delt_thet, EEGfire, EEGamp, EEGmax,
                 EEGmean])

            if np.size(np.where(pd.isnull(EMGamp))[0]) > 0:
                EMGamp[np.isnan(EMGamp)] = 0
            data_addition = np.vstack([data_addition, EMGamp])

            data = np.hstack([data, data_addition])

    df = pd.DataFrame(columns=final_features, data=data.T)
    Sleep_Model = SWS_utils.update_sleep_model(model_dir, mod_name, df)
    jobname, x_features = SWS_utils.load_joblib(final_features, emg_flag, mod_name)
    Sleep_Model = Sleep_Model.drop(index=np.where(Sleep_Model['EMG'].isin(['nan']))[0])
    SWS_utils.retrain_model(Sleep_Model, x_features, model_dir, jobname)


if __name__ == "__main__":
    # execute only if run as a script
    filename_sw = '/Users/annzhou/research/neuroscience/ChenLab_Sleep_Scoring/Score_Settings.json'
    with open(filename_sw, 'r') as f:
        d = json.load(f)

    extracted_dir = str(d['savedir'])
    epochlen = int(d['epochlen'])
    fsd = int(d['fsd'])
    emg_flag = int(d['emg'])
    vid_flag = int(d['vid'])
    model_dir = str(d['model_dir'])
    animal = str(d['animal'])
    mod_name = str(d['mod_name'])

    train_first_model(extracted_dir, epochlen, fsd, emg_flag, animal, model_dir, mod_name)