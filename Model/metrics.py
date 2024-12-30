import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import collections


def cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = cm.ravel()

    return cm, tp, fp, fn, tn

def cm_plot(y_true, y_pred, cm, savepath, marker):
    num_class = 2
    cm_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['HC', 'RBD'], cmap = plt.cm.Blues,  normalize='pred', colorbar = False, include_values=False)

    # get the colormap of the confusion matrix
    cmap = cm_disp.im_.cmap
    no_ = cm_disp.im_.norm

    for i in range(num_class):
        for j in range(num_class):
            count = cm[i, j]
            percent = (count/np.sum(cm[:,j]))*100

            # adapt the color of the text w.r.t the background color
            color_val = cm_disp.im_.get_array()[i, j]
            color = cmap(no_(color_val))
            txt_col = 'white' if np.mean(color[:3]) <= 0.5 else 'black'
            cm_disp.ax_.text(j, i, f'{count}\n{percent:.2f}%', ha='center', va='center', color = txt_col, fontsize=12)
            cm_disp.ax_.set(title=f"Confusion matrix for {marker}", xlabel = 'True labels', ylabel = 'Predicted labels')
            plt.savefig(savepath+f'\\cm_{marker}.png')
            plt.savefig(savepath+f'\\cm_{marker}.svg')



def accuracy(y_true, y_pred):
    _, tp, fp, fn, tn = cm(y_true, y_pred)
    acc = (tp + tn) / (tp + fn + tn + fp)
    return acc


def recall(y_true, y_pred):
    _, tp, _, fn, _ = cm(y_true, y_pred)
    recall = tp / (tp + fn)
    return recall

def specificity(y_true, y_pred):
    _, _, fp, _, tn  = cm(y_true, y_pred)
    specificity = tn / (tn + fp)
    return specificity

def precision(y_true, y_pred):
    _, tp, fp, _, _  = cm(y_true, y_pred)
    precision = tp / (tp + fp)
    return precision

def f1_score(y_true, y_pred):
    recall_s = recall(y_true, y_pred)
    precision_s = precision(y_true, y_pred)
    f1_score = 2 * (recall_s * precision_s) / (recall_s + precision_s)
    return f1_score

def track_results(y_true, y_pred, y_id):
    correct_pred_id = np.where(y_true == y_pred)[0]
    wrong_pred_id = np.where(y_true != y_pred)[0]

    return collections.Counter(y_id), collections.Counter(y_id[correct_pred_id]), collections.Counter(y_id[wrong_pred_id])

def METRICS(y_true, y_pred, y_id, savepath, marker):

    cm_, _, _, _, _ = cm(y_true, y_pred)
    cm_plot(y_true, y_pred, cm_, savepath, marker)
    tot_rem_subj, cor_rem_subj, wr_rem_subj = track_results(y_true, y_pred, y_id)

    with open(savepath+f'\\{marker}.txt', 'w') as f:
        f.write(f'accuracy: {accuracy(y_true, y_pred)} \n')
        f.write(f'recall: {recall(y_true, y_pred)} \n')
        f.write(f'specificity: {specificity(y_true, y_pred)} \n')
        f.write(f'precision: {precision(y_true, y_pred)} \n')
        f.write(f'f1_score: {f1_score(y_true, y_pred)}\n')
        f.write('\n')
        f.write(f'Total number of REM epochs per each subject: {tot_rem_subj}\n')
        f.write(f'Total number of REM epochs per each subject (correct): {cor_rem_subj}\n')
        f.write(f'Total number of REM epochs per each subject (missclassified): {wr_rem_subj}\n')



    print(f'{marker} set: accuracy {accuracy(y_true, y_pred)}, recall {recall(y_true, y_pred)}, specificity {specificity(y_true, y_pred)}, precision {precision(y_true, y_pred)}, f1_score {f1_score(y_true, y_pred)}')



