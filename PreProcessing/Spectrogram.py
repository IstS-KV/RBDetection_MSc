import numpy as np
import glob
import pickle
from scipy.signal import spectrogram, stft
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import pywt
from scipy import signal
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian


def read_pkl(file):
    f = open(file, 'rb')
    return pickle.load(f)

def spectrogram(epoch, fs = 250, window = 'hann', nperseg = 500, noverlap = np.round(500*0.9), nfft = 700, scaling = 'psd'):
    return stft(epoch, fs, window, nperseg, noverlap, nfft)

def wavelet_spec(epoch, scales, wavelet = 'mexh', sampling_period = 1/250):
    return pywt.cwt(epoch, scales, wavelet, sampling_period)

window = gaussian(750, std=8, sym=True)
def spec(epoch, w=window, fs=250):
    SFT = ShortTimeFFT(w, hop=75, fs=fs, scale_to='magnitude')
    Sx = SFT.stft(epoch)
    return Sx

def sgplot2D(sig, overlap, window, Fs, ax=None):
    n_per_segment = 250
    n_overlap = int(np.floor(overlap * n_per_segment))

    if ax == None:
        ax = plt.axes()
    window_samples = signal.get_window(window, n_per_segment)
    _, _, _, im = ax.specgram(sig, Fs=Fs, window=window_samples, NFFT=n_per_segment, noverlap=n_overlap)
    return im

def masking_eeg(input):
    input[113:, :] = input[113:]*0-1
    return input
    
def masking_eog(input):
    input[59:, :] = input[59:]*0-1
    return input

# mask 50-75 Hz because model learns artifacts in the region introduced in the original data
def masking_emg(input):
    input[107:186, :] = input[107:186]*0-1
    return input

def building_spectrogram(tmp_eeg, tmp_emg, tmp_eog, list_x, list_y, list_id, id_init, id, label):

    # apply scaling chaannel-wise and subject-wise
    scale_eeg = RobustScaler().fit_transform(tmp_eeg)
    scale_emg = RobustScaler().fit_transform(tmp_emg)
    scale_eog = RobustScaler().fit_transform(tmp_eog)

    for epoch in range(len(scale_eeg)):
        tmp_eeg_epoch = scale_eeg[epoch, :]
        tmp_emg_epoch = scale_emg[epoch, :]
        tmp_eog_epoch = scale_eog[epoch, :]

        # convert eeg, emg and eog of the current epoch to spectrogram
        f, t, Sxx_eeg = spectrogram(tmp_eeg_epoch)
        _, _, Sxx_emg = spectrogram(tmp_emg_epoch)
        _, _, Sxx_eog = spectrogram(tmp_eog_epoch)

        # extract power information (energy across the frequencies) and trim specs to the same shape (trim parts with no info as much as possible)
        eeg_spec = (np.abs(Sxx_eeg[:252, :])**2)
        emg_spec = np.abs(Sxx_emg[28:280, :])**2
        eog_spec = (np.abs(Sxx_eog[:252, :])**2)

        # mask the frequencies with no useful information in eeg and eog 
        eeg_spec = masking_eeg(eeg_spec)
        eog_spec = masking_eog(eog_spec)
        emg_spec = masking_emg(emg_spec)

        tmp_epoch = np.stack([eeg_spec, emg_spec, eog_spec], axis=2)

        id_new = id_init + id
        list_x.append(tmp_epoch)
        list_id.append(id_new)
        list_y.append(label)

        # plt.figure()
        # plt.imshow(emg_spec, aspect='auto', cmap='jet', extent=[0,30,100,10])
        # plt.gca().invert_yaxis()
        # plt.colorbar()
        # plt.savefig(f'C:\\Users\\MLaccount\\projects\\template-student-project\\SOURCECODE\\PreProcessing\\emg_spec_{epoch}_MASSgaps.svg')
        # plt.show()
    
    return list_x, list_y, list_id, id_new


def Dataset(file_path, list_x, list_y, list_id, id_init, label, set, ind = True):
    file = read_pkl(file_path)

    file_eeg = file[0][f'ori_{set}_eegs']
    file_emg = file[1][f'ori_{set}_emgs']
    file_eog = file[2][f'ori_{set}_eogs']

    ori_subj = file_eeg.keys()

    if len(file)>3:
        # merge dictionaries
        file_eeg_syn = file[3][f'syn_{set}_eegs']
        file_eeg = {**file_eeg, **file_eeg_syn}

        file_emg_syn = file[4][f'syn_{set}_emgs']
        file_emg = {**file_emg, **file_emg_syn}

        file_eog_syn = file[5][f'syn_{set}_eogs']
        file_eog = {**file_eog, **file_eog_syn}

    subj = list(file_eeg.keys())

    assert(len(subj) == len(file_eeg))
    assert(len(subj) == len(file_emg))
    assert(len(subj) == len(file_eog))

    # for aggregated dataset and control group(HCvsRBD original only)
    # here I process all subjects in a list together. They are already grouped as train/val/test or hc/rbd. 
    if set == 'train' or set == 'val' or set == 'test':
        for id, name in enumerate(subj):
            tmp_eeg = file_eeg[name]
            tmp_emg = file_emg[name]
            tmp_eog = file_eog[name]

            list_x, list_y, list_id, id_new = building_spectrogram(tmp_eeg, tmp_emg, tmp_eog, list_x, list_y, list_id, id_init, id, label)    
            print(name, id_new)

        return id_new

    # for baseline dataset HCvsRBD (original and synthetic)
    # here I group together original and synthetic data that belong to one subject and then store them in groups, easier to implemet for cross-validation

    # with current code CAP HC would be processed under this condition, but to generate dataset with HC and RBD subj the code above was executed
    if set == 'hc' or set == 'rbd':

        glob_x = []
        glob_y = []
        glob_id =[]

        id_new = id_init

        for ss in ori_subj:
            temp_group = [x for x in subj if (ss in x) and (len(ss)==len(x) or len(ss+'_syn*')==len(x) and not x.endswith(ss))]
            print(temp_group)

            for id, name in enumerate(temp_group):
                tmp_eeg = file_eeg[name]
                tmp_emg = file_emg[name]
                tmp_eog = file_eog[name]

                list_x, list_y, list_id, id_new = building_spectrogram(tmp_eeg, tmp_emg, tmp_eog, list_x, list_y, list_id, id_new+1, id, label)  

            glob_x.append(list_x)
            glob_y.append(list_y)
            glob_id.append(list_id)
            list_x = []
            list_y = []
            list_id = []            
        
        return glob_x, glob_y, glob_id, id_new


''' An example of using functions to create dataset '''

file_test = input("Enter the location of the .pkl file for the test set")
file_val = input("Enter the location of the .pkl file for the validation set")
file_train = input("Enter the location of the .pkl file for the train set")


test_x = []
test_id = []
test_y = []

val_x = []
val_id = []
val_y = []

train_x = []
train_id = []
train_y = []


# Aggregated dataset
print('Test set')
id_test = Dataset(file_test, test_x, test_y, test_id, id_init = 0, label = 1, set = 'test')
print('id_test for CAP up to ', id_test)
print(len(test_x), len(test_y), len(test_id))
print()

print('Validation set')
id_val = Dataset(file_val, val_x, val_y, val_id, id_init = id_test+1, label = 1, set = 'val')
print('id_val for CAP up to ', id_val)
print(len(val_x), len(val_y), len(val_id))
print()

print('Training set')
id_train = Dataset(file_train, train_x, train_y, train_id, id_init = id_val+1, label = 1, set = 'train')
print('id_train for CAP up to ', id_train)
print(len(train_x), len(train_y), len(train_id))

test_x = np.array(test_x)
test_id = np.array(test_id)
test_y = np.array(test_y)

val_x = np.array(val_x)
val_id = np.array(val_id)
val_y = np.array(val_y)

train_x = np.array(train_x)
train_id = np.array(train_id)
train_y = np.array(train_y)

np.savez('Dataset.npz', test_x = test_x, val_x = val_x, train_x = train_x, test_y = test_y, val_y = val_y, train_y = train_y, test_id = test_id, val_id = val_id, train_id = train_id)
