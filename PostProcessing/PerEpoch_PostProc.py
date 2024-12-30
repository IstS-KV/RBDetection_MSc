# plot ditribution of correctly predicted rem epochs

# Validation subjects were used to find the optimal threshold on ROC curve
# Test subjects were used to evaluate the prformance 

import numpy as np
import matplotlib.pyplot as plt
import ast
import seaborn as sns
import scipy.stats as sps
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import pickle

def count_percent(file, list_, hc_, id_rbd, id_hc, tot_rbd, tot_hc):
    content = file.readlines()    
    total = content[6]
    correct = content[7]
    missclass = content[8]

    start_tot = total.find('Counter(')+len('Counter(')
    total_ = total[start_tot:-2]
    start_cr = correct.find('Counter(')+len('Counter(')
    correct_ = correct[start_cr:-2]
    start_mis = missclass.find('Counter(')+len('Counter(')
    missclass_ = missclass[start_mis:-2]

    total_dic = ast.literal_eval(total_)
    correct_dic = ast.literal_eval(correct_)
    missclass_dic = ast.literal_eval(missclass_)

    for key in total_dic.keys():
        # rbd subjects
        if (key >= 52 and key <82) or (key >= 0 and key <5) or (key >= 122 and key <226) or (key >= 705):
            tot = total_dic[key]
            if key in missclass_dic.keys():
                if key in correct_dic.keys():
                    cor = correct_dic[key]
                else:
                    cor = 0
                    
            p = cor/3 
            list_.append(p)
            id_rbd.append(key)
            tot_rbd.append(tot/3)
        # rest
        else:
            tot = total_dic[key]
            if key in correct_dic.keys():
                if key in missclass_dic.keys():
                    w = missclass_dic[key]
                else:
                    w = 0
        
            p = w/2
            hc_.append(p)
            id_hc.append(key)
            tot_hc.append(tot/2)
            
            
ch = 'eeg emg' # set of input channels
th = 32 # [threshold in minutes], predefined fron ROC analysis on validation set

percent_rbd = [] # contains number of correctly classified epochs per subject [rbs as rbd]
percent_hc = [] # contains number of wrongly classified epochs per subject [hc as rbd]
id_rbd = []
id_hc = []
tot_rbd = []
tot_hc = []

############################### FOR EVALUATION ###########################

# file_tee = open(f"Path\\to\\the\\.txt\\file\\saved\\after\\model\\evaluation (Testing)", 'r') ## 0018
# count_percent(file_tee, percent_rbd, percent_hc, id_rbd, id_hc)

############################### FOR THRESHOLD SEARCH ###########################

file_tee = open(f"Path\\to\\the\\.txt\\file\\saved\\after\\model\\validation (Validation)", 'r') ## 0018
count_percent(file_tee, percent_rbd, percent_hc, id_rbd, id_hc, tot_rbd, tot_hc)

# outputs subjects with number of REM epochs less than threshold value
ix_rbd = [j for j, x in enumerate(tot_rbd) if x < th] 
ix_hc = [j for j, x in enumerate(tot_hc) if x < th]
print('Subjects with less REM epochs than th [rbd]: ', np.array(id_rbd)[ix_rbd])
print('Subjects with less REM epochs than th [hc]: ', np.array(id_hc)[ix_hc])
print()

# remove subject with REM sleep duration lower than a threshold
if len(ix_hc) > 0:
    percent_hc = np.delete(np.array(percent_hc), ix_hc)
    id_hc = np.delete(np.array(id_hc), ix_hc)
    tot_hc = np.delete(np.array(tot_hc), ix_hc)

if len(ix_rbd) > 0:
    percent_rbd = np.delete(np.array(percent_rbd), ix_rbd)
    id_rbd = np.delete(np.array(id_rbd), ix_rbd)
    tot_rbd = np.delete(np.array(tot_rbd), ix_rbd)

# outputs the id of subjects which were predicted falsely 
idx_rbd = [j for j, x in enumerate(percent_rbd) if x < th] 
idx_hc = [j for j, x in enumerate(percent_hc) if x > th]
print('Subjects that were missclassified [rbd]: ', np.array(id_rbd)[idx_rbd])
print('Subjects that were missclassified [hc]: ', np.array(id_hc)[idx_hc])

################################# visualization of REM sleep duration and thresholds #############################

# sort HC and RBD subject by the total duration of REM sleep in descending order
new_order = np.argsort(-np.concatenate([tot_hc, tot_rbd], axis=0))
# indexes of RBD subjects before soriting
rbd_ind = np.arange(len(new_order)-5, len(new_order))
# find new indexes of rbd subjects after sorting
rbd_loc = [j for j, x in enumerate(new_order) if x in rbd_ind]

plt.figure(figsize=(12,8))
b = plt.bar(np.arange(len(tot_hc)+len(tot_rbd)), -np.sort(-np.concatenate([tot_hc, tot_rbd], axis=0)), color='grey', edgecolor='black', alpha=0.5, width=0.8, label = 'HC')
plt.ylabel('REM sleep stage, min')

for j, b0 in enumerate(b):
    if j in rbd_loc:
        b0.set_color('lightgrey')
        b0.set_edgecolor('black')
        b0.set_alpha(0.5)
        b0.set_label('RBD')

length = len(tot_hc)+len(tot_rbd)
plt.plot(np.arange(length), [18]*length, '--', color='red', label = 'EOG')
plt.plot(np.arange(length), [28]*length, '--', color='red', label = 'EMG EOG') # assign diff color
plt.plot(np.arange(length), [25]*length, '--', color='red', label = 'EMG') # assign diff color
plt.plot(np.arange(length), [27]*length, '--', color='red', label = 'EEG EOG') # assign diff color
plt.plot(np.arange(length), [34]*length, '--', color='red', label = 'EEG EMG') # assign diff color
plt.xticks([])
plt.xlabel('Subjects')

plt.legend()
# plt.savefig('Duration distribution.svg')
plt.show()


########################## per-night prediction evaluation ###########################
print()
print(np.min(percent_rbd), np.max(percent_rbd))

th_pool = np.arange(0, np.max(percent_rbd), 1)
se_pool = []
sp_pool = []
g_pool = []
f1_pool = []

for th in th_pool:
    th = np.round(th,2)
    rbd_pred = np.where(percent_rbd > th, 1, 0)
    hc_pred = np.where(percent_hc > th, 1, 0)    

    TN = np.sum(rbd_pred==1)
    FP = np.sum(rbd_pred==0)
    TP = np.sum(hc_pred==0)
    FN = np.sum(hc_pred==1)

    se = TN/(TN+FP)
    sp = TP/(TP+FN)
    re = TP/(TP+FN)
    pre = TP/(TP+FP)

    g = np.sqrt(se*(sp))
    f1 = (2*pre*re)/(pre+re)

    se_pool.append(se)
    sp_pool.append(sp)
    g_pool.append(g)
    f1_pool.append(f1)

    # print(f'Threshold {th}, sensitivity {se}, specificity {sp}')

    # plot confusion matrix for the identified threshold value [possible for both validation and test sets]
    # if th == 68:
    #     cm = np.array([[TP, FN], [FP, TN]])
    #     cm_disp = ConfusionMatrixDisplay(cm, display_labels=['HC','RBD'])
    #     cm_disp.plot(cmap=plt.cm.Greens)    
    #     plt.savefig(f'ConfusinMatrix_{ch}.svg')

ix = np.argmax(g_pool)
# ixx = np.argmax(f1_pool)

print(sp_pool[ix], se_pool[ix])

######################### VISUALIZATION OF ROC PLOT AND OPTIMAL THRESHOLD [apply to validation set] ############################# 

# plt.figure()
# plt.plot(1-np.array(sp_pool), se_pool)
# plt.scatter(1-np.array(sp_pool[ix]), se_pool[ix], label = th_pool[ix])
# plt.xlim([-0.01, 1.01])
# plt.legend()
# plt.show()

# with open(f'roc_curve_{ch}.pkl', 'wb') as f:
#     pickle.dump([1-np.array(sp_pool), se_pool, 1-np.array(sp_pool[ix]), se_pool[ix], th_pool[ix]], f)
    

############################### HC, WESA vs MASS [per-epoch] ##############################

wesa_w = [] # hc from ambulatory study
mass_w = [] # hc from PSG study 

for jj, idd in enumerate(id_hc):
    # WESA
    if idd < 29:
        wesa_w.append(percent_hc[jj])
    # MASS
    else:
        mass_w.append(percent_hc[jj])

wesa_tot = np.sum(wesa_w)
mass_tot = np.sum(mass_w)
hc_tot = np.sum(percent_hc)

wesa = [wesa_tot/hc_tot*100] # percentage of wrongly classified from WESA
mass = [mass_tot/hc_tot*100] # percentage of wrongly classified from MASS

plt.figure()
b0 = plt.bar(np.arange(len(wesa))-0.2,  np.array(wesa), edgecolor='black', width=0.2)
b1 = plt.bar(np.arange(len(mass))+0.2,  np.array(mass), edgecolor='black', width=0.2)
plt.xlim([-0.8,0.8])
plt.ylim([0,100])

for bar in [b0, b1]:
    for b in bar:
        yval = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, yval+0.5,np.round(yval,2), ha='center', va = 'bottom', size='large')

# plt.savefig(f'wesa_mass_percent_{ch}.svg')
        
plt.show()



