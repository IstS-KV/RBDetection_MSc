![CNNFeature.py](./CNNFeature.py) contains the architecture of the model.

![model_compile.py](./model_compile.py) compiles the model. 

![GradCAM.py](./GradCAM.py) contains the GradCam mechanism which is implemented after the last convolutional layer in the architecture. It returns a heatmap for each input instance.  

![metrics.py](./metrics.py) contains the metrics which were used to evaluate the results.

All four files mentioned above need to be in the same folder with training, retraining and testing files. 

![train_model.py](./train_model.py) run to train the model. If necessary, specify the directories of the dataset location and the folder where to store checkpoints.
Change the depth of the input and the names of the channels if you want to experiment with different combinations. 

![test_model.py](./test_model.py) run to test the model on unseen data. Can be used only after training. You need to specify the folder where checkoints are stored and the exact file with the checkpoints. Ensure that the input dimension and the channel type(s) consistent with pretrained checkpoints. 
