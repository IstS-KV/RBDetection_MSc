from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
from scipy.ndimage import gaussian_filter
import cv2

def make_gradcam(input_arr, model):

    grad_model = keras.models.Model(model.inputs, [model.get_layer('tf.__operators__.add_3').output, model.output])

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(input_arr)
        pred_index = tf.argmax(pred[0])

        class_channel = pred[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    grads = gaussian_filter(grads, sigma=1)

    cast_conv_out = tf.cast(conv_out > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_out * cast_grads * grads

    conv_out = conv_out[0]
    guided_grads = guided_grads[0]

    # pooled_grads = tf.reduce_mean(guided_grads, axis=(0,1))
    pooled_grads = guided_grads
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)

    heatmap = cam.numpy()

    # normalize the output of heatmap
    numer = heatmap - np.min(heatmap)
    denom = heatmap.max() - heatmap.min() + 1e-8
    heatmap = numer / denom
    # heatmap = (heatmap*255).astype("uint8")
    
    return heatmap


'''
def save_and_display(inp_arr, heatmap, savepath, alpha = 0.4):
    heatmap = np.uint8(255*heatmap)

    # colormap
    colormap = cv2.COLORMAP_JET
    heatmap_rgb = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    input_rgb = []

    if inp_arr.shape[-1] == 1:
        mask = inp_arr != -1
        valid_inp = inp_arr[mask]
        norm_inp = np.copy(inp_arr)
        norm_inp[mask] = (valid_inp - np.min(valid_inp))/(valid_inp.max() - valid_inp.min() + 1e-8)
        norm_inp[mask] = np.uint8(255*norm_inp[mask])
        input_rgb_1 = cv2.cvtColor(norm_inp, cv2.COLOR_GRAY2RGB)
        input_rgb.append(input_rgb_1)
    elif inp_arr.shape[-1] == 2:
        mask_0 = inp_arr[:,:,0] != -1
        valid_inp_0 = inp_arr[:,:,0][mask_0]
        norm_inp_0 = np.copy(inp_arr[:,:,0])
        # norm_inp_0[mask_0] = (valid_inp_0 - np.min(valid_inp_0))/(valid_inp_0.max() - valid_inp_0.min() + 1e-8)
        # norm_inp_0[~mask_0] = 0

        norm_inp_0 = np.uint8(255*norm_inp_0)
        # input_rgb_1 = cv2.cvtColor(np.uint8(255*norm_inp_0), cv2.COLOR_GRAY2RGB)
        input_rgb_1 = cv2.applyColorMap(norm_inp_0, colormap)
        # input_rgb_1[norm_inp_0==0] = 30

        mask_1 = inp_arr[:,:,1] != -1
        valid_inp_1 = inp_arr[:,:,1][mask_1]
        norm_inp_1 = np.copy(inp_arr[:,:,1])
        # norm_inp_1[mask_1] = (valid_inp_1 - np.min(valid_inp_1))/(valid_inp_1.max() - valid_inp_1.min() + 1e-8)
        norm_inp_1[~mask_1] = 0
        norm_inp_1 = np.uint8(255*norm_inp_1)
        input_rgb_2 = cv2.applyColorMap(norm_inp_1, colormap)
        # input_rgb_2[norm_inp_1==0] = 30
        
        input_rgb.append(input_rgb_1)
        input_rgb.append(input_rgb_2)
    elif inp_arr.shape[-1] == 3:
        input_rgb_1 = cv2.cvtColor(inp_arr[:,:,0], cv2.COLOR_GRAY2RGB)
        input_rgb_2 = cv2.cvtColor(inp_arr[:,:,1], cv2.COLOR_GRAY2RGB)
        input_rgb_3 = cv2.cvtColor(inp_arr[:,:,2], cv2.COLOR_GRAY2RGB)
        input_rgb.append(input_rgb_1)
        input_rgb.append(input_rgb_2)
        input_rgb.append(input_rgb_3)    

    for jj in range(len(input_rgb)):

        plt.figure()
        plt.matshow(input_rgb[jj], cmap='jet')
        plt.gca().xaxis.tick_bottom()
        plt.gca().invert_yaxis()
        # plt.gca().set_box_aspect(30 / 10)
        plt.colorbar(shrink=0.5)
        plt.show()

        output = cv2.addWeighted(input_rgb[jj], alpha, heatmap_rgb, 1 - alpha, 0)
        output = cv2.flip(output, 0)

        plt.figure()
        plt.matshow(output, cmap='jet')
        plt.gca().xaxis.tick_bottom()
        plt.gca().invert_yaxis()
        plt.show()

        # plt.matshow(output, cmap='jet', vmin=255, vmax=0) 
        cv2.imwrite(savepath+f'_{jj}.png', cv2.resize(output, (0,0), fx=3, fy=2))
        # plt.savefig(savepath+f'_{jj}.svg')
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
'''