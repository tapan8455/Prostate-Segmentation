This folder contains the code for the Prostate Segmentation using Deep Learning.For the training of the model the data from PROMISE 12 Challenge is used.
- To download the data for PROMISE12 Challenge go to "https://promise12.grand-challenge.org/Download/"
  - In the 'Download' section, you can download the training data set which consists of three parts. Combine all the training data into one folder.
- To run the code Pythonv3.7 is needed
- The libraries needed to run the code are tensorflow, Keras, Numpt, cv2 and SimpleITK.
- The report of this study is included above. The whole code is divided into four parts: 
  - pre-processing_data.py, which converts the downloaded dataset into a numpy array and contains different algorithms for data preprocessing.
  - deep_learning_model.py, contains the architecture of the model. U-Net is used to train our model. It also contains explanations of the different parameters used.
  - training_model.py, contains the different functions used to train the algorithm.
  - exp_and_results.py, contains all the functions required for training, testing, and evaluating the model, including the training function and various metrics. Additionally, the functions used to run the tests can also be found in this file.
- The code was build using different sources and the references are provided in the code itself.
- ### To ensure that the code runs efficiently, make sure to change the file paths of the training data and other files to your local specific path.
  

