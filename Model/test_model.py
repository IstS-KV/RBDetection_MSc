from absl import flags
import sys
import numpy as np
import os
import tensorflow as tf
from model_compile import model as build_model
from metrics import METRICS
import GradCAM
import matplotlib.pyplot as plt
from tensorflow import keras
import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
from sklearn.utils import shuffle
import pickle

flags.DEFINE_string('load_data_path', 'user\\specified\\path\\to\\load\\the\\dataset', 'load_data path')
flags.DEFINE_string('load_boost_data_path', 'user\\specified\\path\\to\\load\\the\\boost\\dataset', 'load_boost_data path')
flags.DEFINE_string('checkpoint_path', 'user\\specified\\path\\to\\save\\checkpoints', 'checkpoint_path')
flags.DEFINE_integer('batch_size',124, 'batch_size [default 128]')

flags.DEFINE_integer('num_channels', 1, 'num_channels [default 1]')
flags.DEFINE_string('channels_names', 'emg', 'channels_names [default eeg]')
flags.DEFINE_integer('n_class', 2, 'number of output classes [default 2]')

flags.DEFINE_integer('num_epochs', 100, 'number of epochs [default 100]')
flags.DEFINE_integer('num_res_blocks', 0, 'Number of residual blocks [default 1]')
flags.DEFINE_bool('retrain', False, 'retrain [default False]')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

print('Data path: ', FLAGS.load_data_path)
print('Checkpoint path: ', FLAGS.checkpoint_path)
print('Model retrain: ', FLAGS.retrain)
print()

batch_size = FLAGS.batch_size
num_channels = FLAGS.num_channels
num_res_blocks = FLAGS.num_res_blocks
n_class = FLAGS.n_class

# test data
load_checkpoint_path = FLAGS.checkpoint_path + '\\cp-0002.ckptbest_model.h5'
savepath = FLAGS.checkpoint_path + '\\metrics'
data = np.load(FLAGS.load_data_path)
data_boost = np.load(FLAGS.load_boost_data_path)

#################################### DATA PREPARATION ##################################

test_x = data['test_x']
test_y = data['test_y']
test_id = data['test_id']

# print out the shapes
print('Test set (x_shape, y_shape, id_shape): ', test_x.shape, test_y.shape, test_id.shape)
print(f'test_y number of labels 1: {np.count_nonzero(test_y==1)}, 0: {np.count_nonzero(test_y==0)}')
print()

# select the channels
if FLAGS.channels_names == 'eeg':
    test_ch_x = test_x[:,:,:,0:1]
elif FLAGS.channels_names == 'emg':
    test_ch_x = test_x[:,:,:,1:2]
elif FLAGS.channels_names == 'eog':
    test_ch_x = test_x[:,:,:,2:3]
elif FLAGS.channels_names == 'eeg emg':
    test_ch_x = test_x[:,:,:,0:2]
elif FLAGS.channels_names == 'eeg eog':
    test_ch_x = test_x[:,:,:,[0,2]]
elif FLAGS.channels_names == 'emg eog':
    test_ch_x = test_x[:,:,:,1:3]
elif FLAGS.channels_names == 'eeg emg eog':
    test_ch_x = test_x
print('DEBUG shape of input after channel(s) selection (test): ', test_ch_x.shape)
print()

test_dataset = tf.data.Dataset.from_tensor_slices((test_ch_x, test_y))

# check the size of one sample
for i in range(1):
    test_data, test_labels= next(iter(test_dataset))
    print('test data, test label ', test_data.shape, test_labels.shape)

test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

#################################### MODEL ##################################

model = build_model(dropout=True, is_train=False, num_res_blocks=num_res_blocks, batch_size=batch_size, n_class = n_class, num_channels = num_channels)
# model.summary()

model_new = tf.keras.Model(inputs=model.input,
                           outputs=model.output)

model_new.load_weights(load_checkpoint_path)
model.summary()
print()

outputs = model_new.predict(test_ch_x, batch_size=batch_size)
outputs_pred = np.argmax(outputs, axis=1)

#################################### METRICS ##################################

METRICS(outputs_pred, test_y, test_id, savepath, marker = 'Testing')

######################### VISUALIZATION OF UNIFIED ATTENTION MAPS #############################

unique_id = set(test_id)
heatmaps = []

for j in unique_id:
    heatmap_subj = []
    pred_subj = []
    subj_idx = np.where(test_id == j)[0]
    # print(subj_idx)

    for jj in subj_idx:
        w, h = test_ch_x[jj:jj+1, :,:,:].shape[2], test_ch_x[jj:jj+1, :,:,:].shape[1]
        # save all heatmaps belonging to one subject
        heatmap = GradCAM.make_gradcam(test_ch_x[jj:jj+1, :,:,:], model_new)

        ################# visualize and save 30-sec spectrograms and attention maps for them (optional) #########################
        # h_temp = cv2.resize(heatmap, (w,h), interpolation = cv2.INTER_CUBIC)
        # inp_temp = test_ch_x[jj, :,:,0]

        #     plt.figure(figsize=(12,16))
        #     plt.matshow(h_temp[:141], cmap = 'jet', vmin=0, vmax=1) # , extent = [0, 30, 90, 0])
        #     plt.gca().xaxis.tick_bottom()
        #     plt.gca().invert_yaxis()
            # plt.gca().set_box_aspect(30 / 10)
        #     plt.colorbar(shrink=0.5)
            # plt.tight_layout()
        #     plt.title(f'Heatmap for {FLAGS.channels_names}: subj {j}, epoch {jj}, label {test_y[jj]}')
        #     plt.savefig(f'heatmap_subj{j}_epoch {jj}_label{test_y[jj]}.svg')
        #     plt.show()

        #     plt.figure(figsize=(12,16))
        #     plt.matshow(inp_temp[:141], cmap = 'jet', vmin=0, vmax=1) # , extent = [0, 30, 90, 0])
        #     plt.gca().xaxis.tick_bottom()
        #     plt.gca().invert_yaxis()
            # plt.gca().set_box_aspect(30 / 10)
        #     plt.colorbar(shrink=0.5)
            # plt.tight_layout()
        #     plt.title(f'Spectrogram for {FLAGS.channels_names}: subj {j}, epoch {jj}, label {test_y[jj]}')
        #     plt.savefig(f'Spectrogram_subj{j}_epoch{jj}_label{test_y[jj]}.svg')
        #     plt.show()

        # GradCAM.save_and_display(test_ch_x[jj, :,:,:], cv2.resize(heatmap, (w,h), interpolation = cv2.INTER_CUBIC), savepath+f'\\input_heatmap_subj{j}_{FLAGS.channels_names}', alpha=0.6)
        heatmap_subj.append(heatmap)

    heatmap_arr = np.array(heatmap_subj)
    heatmap = np.mean(heatmap_arr, axis=0) # UNIFIED ATTENTION MAP

    heatmaps.append(heatmap)

    cam = cv2.resize(heatmap, (w,h), interpolation = cv2.INTER_CUBIC)
    mask = test_ch_x[jj, :,:,0] == -1
    cam[mask] = 0

    plt.figure(figsize=(12,16))
    plt.matshow(cam[0:59], cmap = 'jet', vmin=0, vmax=1) # , extent = [0, 30, 40, 0])
    plt.gca().xaxis.tick_bottom()
    plt.gca().invert_yaxis()
    # plt.gca().set_box_aspect(30 / 10)
    plt.colorbar(shrink=0.5)
    # plt.tight_layout()
    plt.title(f'Heatmap for {FLAGS.channels_names}: subj {j}, label {test_y[jj]}')
    plt.savefig(savepath+f'\\heatmap_subj{j}.png')
    plt.savefig(savepath+f'\\heatmap_subj{j}.svg')
