Run ![GroundTruth_Preprocessing.ipynb](./GroundTruth_Preprocessing.ipynb) to convert all the ground truth accroding to AASM guidelines (labels 0: wake, 1: n1, 2: n2, 3: n3, 4: REM) for 30-second epochs. Save new ground truth into .txt file.

![CountEpochs.py](./CountEpochs.py) returns the number of epochs for every subject or dataset.

![GenData.py](./GenData.py) contains the functions to extract recordings of interest, preprocess them (filtering, resampling), segment them into 30-second epochs, extract only REM sleep and oversample the minority class (or synthetically increase the number of 30-second epochs and subjects). To execute these functions, use ![DataStruct.ipynb](./DataStruct.ipynb) file, which returns .pkl files for each dataset and data split (train, val, test).

![CorrAnalysis.ipynb](./CorrAnalysis.ipynb) evaluates the quality of synthetic data.

![Spectrogram.py](./Spectrogram.py) uses .pkl files to convert epochs into spectrograms and returns complete datasets (.npz format) which are ready to be used to train, tune or test the models. 

![mat_to_edf.ipynb](./WESA_mat_to_edf.ipynb) is an example of converting .mat file into .edf file. 