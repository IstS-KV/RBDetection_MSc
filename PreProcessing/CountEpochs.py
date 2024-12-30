import glob
import numpy as np
import os
import scipy


def count_epochs(file_path, marker):

    # INPUT
    # file_path --> directory with files that contain recordings
    # marker --> indicates the dataset

    # OUTPUT
    # the number of 30-second epochs dor every sleep stage
    
    total_wake = 0
    total_n1 = 0
    total_n2 = 0
    total_n3 = 0
    total_rem = 0
    total_q = 0
    total_six = 0
    # rems = []

    files_txt = glob.glob(file_path)   
    
    for jj in range(len(files_txt)):
        print(files_txt[jj])
        file = open(files_txt[jj], 'r')
        content = file.read().splitlines()
        # content = [int(x) for x in content]
        total_len = len(content)

        # print(files_txt[jj])

        # count epochs
        wake_epochs = content.count('0')
        n1_epochs = content.count('1')
        n2_epochs = content.count('2')
        n3_epochs = content.count('3')
        rem_epochs = content.count('4')
        # for mass dataset
        q_count = content.count('?')
        # for wesa dataset
        six_count = content.count('6')
        onset_count = 0

        if marker == 'mass':
            onset_count = 1
        
        # update the number
        total_wake = total_wake + wake_epochs
        total_n1 = total_n1 + n1_epochs
        total_n2 = total_n2 + n2_epochs
        total_n3 = total_n3 + n3_epochs
        total_rem = total_rem + rem_epochs
        total_q = total_q + q_count
        total_six = total_six + six_count

        # per subject
        print(f'{file}: wake, n1, n2, n3, rem, quest_mark, six', wake_epochs, n1_epochs, n2_epochs, n3_epochs, rem_epochs, q_count, six_count)
        print('')
        # rems.append(rem_epochs)

        assert(total_len == wake_epochs + n1_epochs + n2_epochs + n3_epochs + rem_epochs + q_count + six_count + onset_count)  
        
        file.close()  
    print(f'{marker}: wake, n1, n2, n3, rem, quest_mark, six', total_wake, total_n1, total_n2, total_n3, total_rem, total_q, total_six)

    return total_wake, total_n1, total_n2, total_n3, total_rem, total_q, total_six # rems      

