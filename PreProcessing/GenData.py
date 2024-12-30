
import re
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from scipy.signal import iirnotch
from scipy.signal import butter, lfilter
from scipy.io import loadmat 
from scipy import signal
from scipy.signal import resample_poly
from scipy.interpolate import interp1d

def resample_signal(original_signal, original_fs, new_fs):
 
    # Calculate the new number of samples
    original_time = np.linspace(0, len(original_signal) / original_fs, len(original_signal), endpoint=False)
    new_time = np.linspace(0, len(original_signal) / original_fs, int(len(original_signal) * new_fs / original_fs), endpoint=False)

    # Interpolate 
    interpolator = interp1d(original_time, original_signal, kind='cubic') 
    resampled_signal = interpolator(new_time)

    return resampled_signal

# bandpass filtering
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch_filt(sig, notch_fq, q, fs):
    b_notch, a_notch = iirnotch(notch_fq, q, fs)
    return signal.filtfilt(b_notch, a_notch, sig)

# split the data into train, val and test sets (applied for CAP dataset and MASS SS2,3,5)
def data_split(subj_list, num_epochs_list, name='rbd_subj'):
    
    med = np.median(num_epochs_list)
    rem = []
    
    for i in range(len(num_epochs_list)):
        if num_epochs_list[i] <= med:
            rem.append('0')
        else:
            rem.append('1')
    
    # stratify data based on number of rem epochs
    pddata_ = pd.DataFrame({name: subj_list, 'rem_epochs_num': rem})

    if name == 'rbd_subj':
        tv_, train_ = train_test_split(pddata_, test_size=0.5, random_state=0, stratify=pddata_['rem_epochs_num'])
        vali_, test_ = train_test_split(tv_, test_size=0.5, random_state=42, stratify=tv_['rem_epochs_num'])
        return train_, vali_, test_
    else:
        vali_, test_ = train_test_split(pddata_, test_size=0.42, random_state=42, stratify=pddata_['rem_epochs_num'])
        return vali_, test_


######################## EXTRACT SIGNALS, CAP WESA MASS SS1, SS2, SS3, SS5 ########################################
# frontal EEG (which is available)
# chin EMG 
# EOG (delta of right and left)
# marker indicates different datasets
def extract_eeg_emg_eog(set_, recordings, ground_truth, seg_len = 30, fs = 200, marker = 'CAP'):

    EEGs = []
    EMGs = []
    EOGs = []
    subj = []
    
    br = 0
    for jj in set_:

        if marker == 'CAP':
            ss = jj
            file = [x for x in recordings if ss+'.edf' in x]
            gt_file = [x for x in ground_truth if ss+'.txt' in x]
    
            # read the data
            ch_list = ['Fp2-F4', 'ROC-LOC', 'EMG1-EMG2']
            data = mne.io.read_raw_edf(file[0], include = ch_list, preload=True, verbose=False)
            data.set_channel_types({'EMG1-EMG2': 'emg'})
            old_fs = int(data.info['sfreq'])
            # extract every channel individually 
            eeg = data['Fp2-F4'][0][0]
            emg = data['EMG1-EMG2'][0][0]
            eog = data['ROC-LOC'][0][0]
            # filtering
            eeg_filt =  butter_bandpass_filter(eeg, lowcut = 0.1, highcut = 40, fs = old_fs, order=4)
            emg_filt =  butter_bandpass_filter(emg, lowcut = 10, highcut = 100, fs = old_fs, order=5)
            eog_filt =  butter_bandpass_filter(eog, lowcut = 0.1, highcut = 35, fs = old_fs, order=4)
            # resample the data
            eeg_filt = resample_signal(eeg_filt, old_fs, fs)
            emg_filt = resample_signal(emg_filt, old_fs, fs)
            eog_filt = resample_signal(eog_filt, old_fs, fs)
            
        elif marker == 'WESA':
            idx = jj.find('WESA_EEG')
            ss = jj[idx:-4]
            
            file = loadmat(jj)
            print(ss)
            gt_file = [x for x in ground_truth if ss in x]
            print(gt_file)
            fs_hc_data = file['sampling_rate']

            eeg = file['FpzA2'][0]
            emg = file['EMG'][0]
            eog = file['EOGL'][0] - file['EOGR'][0]

            eeg_filt =  butter_bandpass_filter(eeg, lowcut = 0.1, highcut = 40, fs = fs, order=4)
            emg_filt =  butter_bandpass_filter(emg, lowcut = 10, highcut = 100, fs = fs, order=5)
            emg_filt =  notch_filt(emg_filt, 50, 100, fs)
            eog_filt =  butter_bandpass_filter(eog, lowcut = 0.1, highcut = 35, fs = fs, order=4)

        elif marker == 'MASS':
            file = jj
            idx = file.find('MASS\\') + len('MASS\\') + len('SS*\\')
            ss = file[idx:-7]
            gt_file = [x for x in ground_truth if ss+'Base.txt' in x]

            # read the data
            ch_list = ['EEG Fz-CLE', 'EEG Fz-LER', 'EEG Fp2-LER', 'EEG Fp2-CLE', 'EOG Left Horiz', 'EOG Right Horiz', \
                       'EMG Chin', 'EMG Chin1', 'EMG Chin2', 'EMG Chin3']
            data = mne.io.read_raw_edf(file, include = ch_list, preload=True, verbose=False)
            old_fs = int(data.info['sfreq'])

            # handling available channels in different subsets (EEG and EMG)
            eeg_ch = 'none'
            emg2 = 0
            for jjj in data.info['ch_names']:
                # I give priority to Fz
                if 'EEG Fz' in jjj:
                    eeg_ch = jjj
                else:
                    if 'EEG' in jjj and eeg_ch == 'none':
                        eeg_ch = jjj

            if 'SS1' in file or 'SS3' in file:
                emg_ch = 'EMG Chin3'
            elif 'SS2' in file or 'SS4' in file :
                emg_ch = 'EMG Chin'
            elif 'SS5' in file:
                emg_ch = 'EMG Chin1'
                emg_ch2 = 'EMG Chin2'
                data.set_channel_types({emg_ch2: 'emg'})
                emg2 = data[emg_ch2][0][0]
            
            # set the channel types       
            data.set_channel_types({emg_ch: 'emg'})
            data.set_channel_types({'EOG Left Horiz': 'eog'})
            data.set_channel_types({'EOG Right Horiz': 'eog'})
            
            # extract every channel individually
            eeg = data[eeg_ch][0][0]
            emg = data[emg_ch][0][0] - emg2
            eog = data['EOG Right Horiz'][0][0] - data['EOG Left Horiz'][0][0]
            # reset the name
            eeg_ch = 'none'

            # filtering
            eeg_filt =  butter_bandpass_filter(eeg, lowcut = 0.1, highcut = 40, fs = old_fs, order=4)
            emg_filt =  butter_bandpass_filter(emg, lowcut = 10, highcut = 100, fs = old_fs, order=5)
            emg_filt = notch_filt(emg_filt, 60, 100, fs)
            eog_filt =  butter_bandpass_filter(eog, lowcut = 0.1, highcut = 35, fs = old_fs, order=4)

            # resample the data
            eeg_filt = resample_signal(eeg_filt, old_fs, fs)
            emg_filt = resample_signal(emg_filt, old_fs, fs)
            eog_filt = resample_signal(eog_filt, old_fs, fs)
             
        # read the ground truth
        with open(gt_file[0], 'r') as ff:
            if marker == 'MASS':
                y = [line.rstrip() for line in ff]
                # add '?' label for the onset epochs
                onset_time = np.round(float(re.findall(r'\d+\.\d+', y[0])[0]))
                onset_epochs =  int(onset_time//2) # int(np.round(onset_time/30))
                y = ['?']*onset_epochs + y[1:]
                y = [int(x) for x in y if x != '?']
            else:
                y = [int(line.rstrip()) for line in ff]
            
        # count number of 30 seconds segments with new resampling in each signal
        seg_num_eeg = np.round(len(eeg_filt)//seg_len//fs)
        seg_num_emg = np.round(len(emg_filt)//seg_len//fs)
        seg_num_eog = np.round(len(eog_filt)//seg_len//fs)

        # compare the number of labeled segment with available number of epochs in every signal. find the minimum (usually the number of expert's labels)
        # cut all tll signals to that number of segments (num of segments in every signal should be same within each subj)
        num_eeg = min(len(y), seg_num_eeg)
        num_emg = min(len(y), seg_num_emg)
        num_eog = min(len(y), seg_num_eog)
        num_min = min(num_eeg, num_emg, num_eog)
        # estimate new length of the signal
        new_len = num_min*seg_len*fs 
        eeg = eeg_filt[:new_len]
        emg = emg_filt[:new_len]
        eog = eog_filt[:new_len]
        y = y[:num_min]


        if marker == 'CAP':
            # THIS IS TO PROCESS ALL SIGNAL FOR 30 SEC EPOCHS WITH OVERLAP [15 sec]

            # find location of REM epochs
            rem_idx = [i for i, x in enumerate(y) if x==4]
            n2 = [i for i, x in enumerate(y) if x==2]
            # print(len(n2))
            ind = [0]
            [ind.append(i) for i, x in enumerate(np.diff(rem_idx)) if x > 1]
            ind.append(len(rem_idx))

            # save consecutive rem episodes
            consec_rem = []
            for jj in range(1, len(ind)):
                if jj == 1:
                    consec_rem.append(rem_idx[ind[jj-1]:ind[jj]+1])
                else:
                    consec_rem.append(rem_idx[ind[jj-1]+1:ind[jj]+1])


            overlap = 15
            sig_bins = seg_len*fs
            o_bins = overlap*fs
            step = sig_bins-o_bins
            eeg_new = []
            eog_new = []
            emg_new = []
            if len(consec_rem)>0:
                for j in range(len(consec_rem)):
                    temp_episode = consec_rem[j]
                    # add 15 seconds before rem episode and after rem episode
                    start_ep = temp_episode[0]*seg_len*fs - seg_len//2*fs
                    end_ep = temp_episode[-1]*seg_len*fs + seg_len//2*fs+1

                    # extract region of interest for each signal
                    temp_eeg =  eeg[start_ep:end_ep]
                    temp_eog =  eog[start_ep:end_ep]
                    temp_emg =  emg[start_ep:end_ep]

                    # create 30 second epochs with 5 second overlap
                    epoch_num = (len(temp_eeg)-o_bins)//(seg_len*fs-o_bins)

                    for e in range(epoch_num):
                        eeg_epoch = temp_eeg[e*step:e*step+seg_len*fs]
                        eog_epoch = temp_eog[e*step:e*step+seg_len*fs]
                        emg_epoch = temp_emg[e*step:e*step+seg_len*fs]

                        eeg_new.append(eeg_epoch)
                        eog_new.append(eog_epoch)
                        emg_new.append(emg_epoch)

            # convert to arrays
            eeg = np.array(eeg_new)     
            eog = np.array(eog_new) 
            emg = np.array(emg_new)           

            print(eeg.shape)
        else: 
            # THIS IS TO PROCESS ALL SIGNAL FOR 30 SEC EPOCHS WITH NO OVERLAP
                    
            # reshape to a proper format (num of segment, len of the segment) 
            eeg = eeg.reshape([num_min, seg_len*fs])
            emg = emg.reshape([num_min, seg_len*fs])
            eog = eog.reshape([num_min, seg_len*fs])
            # use ground truth to extract and to list only REM epochs (label 4)
            index = []
            for ii, val in enumerate(y):
               if val == 4: 
                   index.append(ii)
            eeg = eeg[index]
            emg = emg[index]
            eog = eog[index]

        # add shuffling to destroy the queue
        eeg, emg, eog = shuffle(eeg, emg, eog, random_state = 1)

        EEGs.append(eeg)
        EMGs.append(emg)
        EOGs.append(eog)
        subj.append(ss)

        print(f'Subject {ss} complete')

    return EEGs, EMGs, EOGs, subj

######################## EXTRACT SIGNALS, ONLY FOR SS4 ########################################
# overlap is not implemented

def extract_eeg_emg_eog_SS4(set_, ground_truth, seg_len = 30, fs = 200):

    EMGs = []
    EOGs = []
    subj = []
    
    br = 0
    for jj in set_:
        file = jj
        idx = file.find('MASS\\') + len('MASS\\') + len('SS*\\')
        ss = file[idx:-7]
        gt_file = [x for x in ground_truth if ss+'Base.txt' in x]
        
        ch_list = ['EOG Left Horiz', 'EOG Right Horiz', 'EMG Chin']
        data = mne.io.read_raw_edf(file, include = ch_list, preload=True, verbose=False)
        data_resample = data.copy().resample(sfreq=fs)
        
        # resample the data        
        data_resample.set_channel_types({'EMG Chin': 'emg'})
        data_resample.set_channel_types({'EOG Left Horiz': 'eog'})
        data_resample.set_channel_types({'EOG Right Horiz': 'eog'})
        
        # extract every channel individually
        emg = data_resample['EMG Chin'][0][0] 
        eog = data_resample['EOG Right Horiz'][0][0] - data_resample['EOG Left Horiz'][0][0]

        # filtering
        emg_filt =  butter_bandpass_filter(emg, lowcut = 10, highcut = 100, fs = fs, order=5)
        eog_filt =  butter_bandpass_filter(eog, lowcut = 0.1, highcut = 35, fs = fs, order=5)
             
        # read the ground truth
        with open(gt_file[0], 'r') as ff:
            y = [line.rstrip() for line in ff]
            # add '?' label for the onset epochs
            onset_time = np.round(float(re.findall(r'\d+\.\d+', y[0])[0]))
            onset_epochs = np.round(onset_time/30)
            y = ['?']*onset_epochs + y[1:]
            y = [int(x) for x in y if x != '?']
            
        # count number of 30 seconds segments with new resampling in each signal
        seg_num_emg = np.round(len(emg_filt)//seg_len//fs)
        seg_num_eog = np.round(len(eog_filt)//seg_len//fs)

        # compare the number of labeled segment with available number of epochs in every signal. find the minimum (usually the number of expert's labels)
        # cut all tll signals to that number of segments (num of segments in every signal should be same within each subj)
        num_emg = min(len(y), seg_num_emg)
        num_eog = min(len(y), seg_num_eog)
        num_min = min(num_emg, num_eog)
        # estimate new length of the signal
        new_len = num_min*seg_len*fs 
        emg = emg_filt[:new_len]
        eog = eog_filt[:new_len]
        y = y[:num_min]
        # reshape to a proper format (num of segment, len of the segment)
        emg = emg.reshape([num_min, seg_len*fs])
        eog = eog.reshape([num_min, seg_len*fs])
        # use ground truth to extract and to list only REM epochs (label 4)
        index = []
        for ii, val in enumerate(y):
            if val == 4:
                index.append(ii)
        emg = emg[index]
        eog = eog[index]

        # add shuffling to destroy the queue
        emg, eog = shuffle(emg, eog, random_state = 1)

        EMGs.append(emg)
        EOGs.append(eog)
        subj.append(ss)

    return EMGs, EOGs, subj

########################### SMOTE to balance RBD and HC classes ######################################
# APPLIED ONLY TO RBD SUBJECTS OF TRAIN AND VAL SUBSETS
# 1. select an original subject
# 2. generate multiple new synthetic epochs 
# 3. select a random number between between max and min possible number of REM epochs in respective class
# 4. Assigns the number (from the previous step) of epochs to generate 1 synthetic subject
# 5. Repeat 3-4 until all epochs will be split (in the last case, if the number of lest epochs is less than the selected number, the algo checks if the remaining number of epochs 
#    is higher than the min value. If yes, it uses this epochs to generate one more subject. If not, the algo moves to the next subject)
# 6. Repeat for every subject

def oversample(channel_data_list, syn_num = 9000, random_state=42):

    subj_num = len(channel_data_list)
    # num_rem = []
    fake_labels = []
    target = {}
    total_syn_num = 0

    for jj in range(subj_num):        
        fake_labels.append(np.expand_dims(np.array([jj]*len(channel_data_list[jj])), axis=1))
        target[jj] = syn_num
        total_syn_num = total_syn_num + syn_num

    data = np.vstack(channel_data_list)
    fl = np.vstack(fake_labels)

    # to diffirintiate between original and synthetic data --> SMOTE RETURNS THE DATA IN THE ORDER: ORIGINAL, SYNTHETIC
    indicator = ['original'] * len(fl) + ['synthetic'] * (total_syn_num)

    smote = SMOTE(sampling_strategy = target, random_state = random_state)
    X_res, y_res = smote.fit_resample(data, fl)

    return X_res, y_res, indicator 

# create synthetic subjects by dividing synthetic epochs (random number of epochs in predefined range) into groups
def syn_subj_gen(num, X_res, y_res, index_syns, global_min, random_nums):
    idx = np.where(y_res == num)[0]
    idx_syn = list(set(index_syns) & set(idx))
    syn = []
    iii = 0
    
    while len(idx_syn) > 0:
        rnd_num = random_nums[iii]
        if rnd_num <= len(idx_syn):
            idx_syn_tmp = idx_syn[:rnd_num]
            syn_subj_tmp = X_res[idx_syn_tmp, :]
            syn.append(syn_subj_tmp)
            idx_syn = idx_syn[rnd_num+1:]
        else:
            if len(idx_syn) >= global_min:
                idx_syn_tmp = idx_syn
                syn_subj_tmp = X_res[idx_syn_tmp, :]
                syn.append(syn_subj_tmp)
                idx_syn = []
            else:
                idx_syn = []
        iii = iii + 1
    return syn, idx

# post processing of the synthetic segments to find original and synthetic data
def group_data(indicator, X_res, y_res, subj_list, global_min, global_max, mode):
    idx_ori = [idx for idx, state in enumerate(indicator) if state == 'original']
    idx_syn = [idx for idx, state in enumerate(indicator) if state == 'synthetic']

    if mode == 'train':
        size = 18
        seed = 0
    elif mode == 'val':
        size = 10
        seed = 42
    
    yy = len(np.unique(y_res))
    ori = {}
    syn = {}
    # find every subject        
    for jj in range(yy):
        np.random.seed(seed + jj)
        random_nums = np.random.randint(global_min, global_max, size = size)
        # print(random_nums)
        syn[subj_list[jj]], idx_subj = syn_subj_gen(jj, X_res, y_res, idx_syn, global_min, random_nums)
        idx_ori_tmp = list(set(idx_ori) & set(idx_subj))
        ori[subj_list[jj]] = (X_res[idx_ori_tmp, :])

    return ori, syn