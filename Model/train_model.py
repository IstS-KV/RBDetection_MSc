import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from model_compile import model as build_model
from absl import flags
from sklearn.utils import shuffle
from metrics import METRICS
import GradCAM
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import metrics
import pickle
import cv2


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.set_logical_device_configuration(
            #     gpu, 
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e: 
        print(e)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

flags.DEFINE_string('load_data_path', 'user\\specified\\path\\to\\load\\the\\dataset', 'load_data path')
flags.DEFINE_string('save_checkpoint_path', 'user\\specified\\path\\to\\save\\checkpoints', 'save_checkpoint_path')
flags.DEFINE_integer('batch_size', 128, 'batch_size [default 128]')
flags.DEFINE_integer('n_class', 2, 'number of output classes [default 2]')
flags.DEFINE_integer('num_channels', 1, 'num_channels [default 1]')
flags.DEFINE_integer('num_epochs', 100, 'number of epochs [default 100]')
flags.DEFINE_integer('shuffle_buffer_size', 1000, 'suffle_buffer-size [default 1000]')
flags.DEFINE_string('channels_names', 'eeg', 'channels_names [default eeg]')
flags.DEFINE_integer('num_res_blocks', 0, 'Number of residual blocks [default 0]')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

print('Data path: ', FLAGS.load_data_path)
print('Save checkpoint path: ', FLAGS.save_checkpoint_path)
print()

tf.io.gfile.makedirs(FLAGS.save_checkpoint_path)

batch_size = FLAGS.batch_size
n_class = FLAGS.n_class
num_channels = FLAGS.num_channels
num_epochs = FLAGS.num_epochs
shuffle_buffer_size = FLAGS.shuffle_buffer_size
num_res_blocks = FLAGS.num_res_blocks

data = np.load(FLAGS.load_data_path)

#################################### DATA PREPARATION ##################################

train_x = data['train_x']
train_y = data['train_y']
train_id = data['train_id']

val_x = data['val_x']
val_y = data['val_y']
val_id = data['val_id']

n0_train = np.count_nonzero(train_y==0)
n1_train = np.count_nonzero(train_y==1)
ntotal_train = n0_train + n1_train
weight_for_0_train = (1/n0_train)*(ntotal_train)/2.0
weight_for_1_train = (1/n1_train)*(ntotal_train)/2.0
class_weight_0_1 = {0: weight_for_0_train, 1: weight_for_1_train}

print('Training set (x_shape, y_shape, id_shape): ', train_x.shape, train_y.shape, train_id.shape)
print('Validation set (x_shape, y_shape, id_shape): ', val_x.shape, val_y.shape, val_id.shape)
print()
print('Defined parameters:')
print('Batch size: ', batch_size)
print('Number of training epochs: ', num_epochs)
print('Suffle buffer size: ', shuffle_buffer_size)
print()
print('Number of output classes: ', 2)
print(f'Weight for class 0 is {weight_for_0_train}, and weight for class 1 is {weight_for_1_train}')
print('Number of channels: ', num_channels)
print('Input channels are from: ', FLAGS.channels_names)

train_x, train_y, train_id = shuffle(train_x, train_y, train_id, random_state=42)
val_x, val_y, val_id = shuffle(val_x, val_y, val_id, random_state=42)

# select the channels
if FLAGS.channels_names == 'eeg':
    train_ch_x = train_x[:,:,:,0:1]
    val_ch_x = val_x[:,:,:,0:1]
elif FLAGS.channels_names == 'emg':
    train_ch_x = train_x[:,:,:,1:2]
    val_ch_x = val_x[:,:,:,1:2]
elif FLAGS.channels_names == 'eog':
    train_ch_x = train_x[:,:,:,2:3]
    val_ch_x = val_x[:,:,:,2:3]
elif FLAGS.channels_names == 'eeg emg':
    train_ch_x = train_x[:,:,:,0:2]
    val_ch_x = val_x[:,:,:,0:2]
elif FLAGS.channels_names == 'eeg eog':
    train_ch_x = train_x[:,:,:,[0,2]]
    val_ch_x = val_x[:,:,:,[0,2]]
elif FLAGS.channels_names == 'emg eog':
    train_ch_x = train_x[:,:,:,1:3]
    val_ch_x = val_x[:,:,:,1:3]
elif FLAGS.channels_names == 'eeg emg eog':
    train_ch_x = train_x
    val_ch_x = val_x

print('DEBUG shape of input after channel(s) selection (training, validation): ', train_ch_x.shape, val_ch_x.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((train_ch_x, train_y))
val_dataset = tf.data.Dataset.from_tensor_slices((val_ch_x, val_y))

print()
print('Number of HC vs RBD epochs in training set: ', np.sum(train_y == 0), np.sum(train_y == 1))
print('Number of HC vs RBD epochs in validation set: ', np.sum(val_y == 0), np.sum(val_y == 1))
print()


# check the size of one sample
for i in range(1):
    val_data, val_labels= next(iter(val_dataset))
    print('val data, val label, val id: ', val_data.shape, val_labels.shape)
    print()

train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

#################################### MODEL ##################################

model = build_model(dropout=True, is_train=True, num_res_blocks=num_res_blocks, batch_size=batch_size, n_class = n_class, num_channels = num_channels)

# save checkpoints
checkpoint_path = FLAGS.save_checkpoint_path + '/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

 # create callback to save model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    period=5)
# early stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True)
best_model_cp = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path+'best_model.h5',
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True,
    verbose=1)

# model save weights in checkpoint_path format
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(train_dataset,
            validation_data=val_dataset,
            epochs=num_epochs,
            callbacks=[cp_callback, early_stopping_callback, best_model_cp],
                       class_weight=class_weight_0_1)

model.summary()
print()

# plot accuracy and loss curves
tf.io.gfile.makedirs(FLAGS.save_checkpoint_path + '\\metrics')
savepath = FLAGS.save_checkpoint_path + '\\metrics' 

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(16,16))
plt.subplot(211)
plt.plot(train_acc[:-10], label='Train Accuracy')
plt.plot(val_acc[:-10], label = 'Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.legend()

plt.subplot(212)
plt.plot(train_loss[:-10], label='Train loss')
plt.plot(val_loss[:-10], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig(savepath+'\\metrics.png')
with open(savepath+'\\train_proc.pkl', 'wb')as f:
    pickle.dump([train_acc, val_acc, train_loss, val_loss], f)

train_pred = model.predict(train_ch_x, batch_size=batch_size)
y_train_pred = np.argmax(train_pred, axis=1)

val_pred = model.predict(val_ch_x, batch_size=batch_size)
y_val_pred = np.argmax(val_pred, axis=1)

#################################### METRICS ##################################

METRICS(y_train_pred, train_y, train_id, savepath, marker = 'Training')
print()
METRICS(y_val_pred, val_y, val_id, savepath, marker = 'Validation')

######################### VISUALIZATION OF UNIFIED ATTENTION MAPS [validation set] ############################# 

unique_id = set(val_id)
heatmaps = []

for j in unique_id:
    heatmap_subj = []
    subj_idx = np.where(val_id == j)[0]

    for jj in subj_idx:

        # save all heatmaps belonging to one subject 
        heatmap = GradCAM.make_gradcam(val_ch_x[jj:jj+1, :,:,:], model)
        heatmap_subj.append(heatmap)

    # generate a unified attention map by averaging all 30-second attention maps
    heatmap_arr = np.array(heatmap_subj)
    heatmap = np.mean(heatmap_arr, axis=0)

    heatmaps.append(heatmap)

    # visualization of a unified attention map 
    w, h = val_ch_x[jj:jj+1, :,:,:].shape[2], val_ch_x[jj:jj+1, :,:,:].shape[1]
    heatmap = np.resize(heatmap, (val_ch_x[jj:jj+1, :,:,:].shape[2], val_ch_x[jj:jj+1, :,:,:].shape[2]))
    cam = cv2.resize(heatmap, (w,h), interpolation = cv2.INTER_CUBIC)

    plt.figure(figsize=(12,16))
    plt.matshow(cam, cmap = 'jet', vmin=0, vmax=1) # ), extent = [0, 30, 90, 0])
    plt.gca().xaxis.tick_bottom()
    plt.gca().invert_yaxis()
    # plt.gca().set_box_aspect(30 / 10)
    plt.colorbar(shrink=0.5)
    plt.tight_layout()
    plt.title(f'Heatmap for {FLAGS.channels_names}: subj {j}, label {val_y[jj]}')
    plt.savefig(savepath+f'\\heatmap_subj{j}.png')
