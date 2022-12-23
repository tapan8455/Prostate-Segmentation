# the code preprocess the images from PROMISE12 challenge and convert it into numpy arrays
# data into numpy arrays
import os
import cv2
import numpy as np
import SimpleITK as sitk 
import random

# The method iteslf was inspired by : https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py
# So the code is heavily inspired by the above source, but the functions are rebuilt according to the need for our project.


def elastic_transform(image, x=None, y=None, alpha=256*3, sigma=256*0.07):
    # inpired by https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a#file-elastic_transform-py and https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/metrics.py 
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    shape = image.shape
    blur_size = int(4*sigma) | 1
    dx = cv2.GaussianBlur((np.random.rand(shape[0],shape[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)* alpha
    dy = cv2.GaussianBlur((np.random.rand(shape[0],shape[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)* alpha

    if (x is None) or (y is None):
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    map_x = (x+dx).astype('float32')
    map_y = (y+dy).astype('float32')

    return cv2.remap(image.astype('float32'), map_y,  map_x, interpolation=cv2.INTER_NEAREST).reshape(shape)

def resize_image(imgs, img_rows, img_cols):
    # As discusses in report MRI slices are of different shape, so we reshape it to a same size that is  256*256.
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        new_imgs[mm] = cv2.resize( img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)
    return new_imgs


def generate_val_list():
    randomlist = []
    for i in range(0,5):
        # 41 cases because 8 cases are left for testing
        n = random.randint(0,41)
        randomlist.append(n)
    return randomlist


def save_to_array(img_list, type_of_set):
    fileList =  os.listdir('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/')
    # fileList is the location of the training data.
    fileList = list(filter(lambda x: '.mhd' in x, fileList))
    print(fileList)
    fileList.sort()

    images = []
    masks = []

    filtered = filter(lambda x: any(str(ff).zfill(2) in x for ff in img_list), fileList)

    for filename in filtered:
        if filename[0] != "." :
            itkimage = sitk.ReadImage('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/'+filename)
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs= resize_image(imgs, 256, 256)
                masks.append( imgs )

            else:
                imgs = resize_image(imgs, 256, 256)
                images.append(imgs)

    images = np.concatenate( images , axis=0 ).reshape(-1, 256, 256, 1)
    masks = np.concatenate(masks, axis=0).reshape(-1, 256, 256, 1)
    masks = masks.astype(int)

    images = curv_denoising(images)
    if type_of_set == "train" : 
        mu = np.mean(images) 
        sigma = np.std(images)
        images = (images - mu)/sigma
        np.save('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/X_train.npy', images)
        np.save('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/y_train.npy', masks)
    else : 
        images = (images - mu)/sigma
        np.save('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/X_val.npy', images)
        np.save('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/y_val.npy', masks)


def curv_denoising(imgs):
    # inspired by https://github.com/mirzaevinom/promise12_segmentation/blob/master/codes/augmenters.py#L31
    # define the parametres
    timeStep=0.186
    numberOfIterations=5

    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                        timeStep=timeStep,
                                        numberOfIterations=numberOfIterations)

        imgs[mm] = sitk.GetArrayFromImage(img)


    return imgs


def convert_img_to_array():
    testing_list = [42, 43, 44, 45, 46, 47, 48, 49]
    val_list = generate_val_list()
    train_list = list( set(range(42)) - set(val_list))

    save_to_array(train_list, "train")
    save_to_array(testing_list, "test")
    save_to_array(val_list, "val")
    #save_to_array(test1_list,"test1")


def load_data():

    X_train = np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/X_train.npy')
    y_train = np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/y_train.npy')
    X_val = np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/X_val.npy')
    y_val = np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/y_val.npy')
    X_test = np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/X_test.npy')
    y_test = np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/y_test.npy')

    return X_train, y_train, X_val, y_val, X_test, y_test

# I load only training sometimes, when I don't want to test for expample, so 

def load_only_training():
    X_train = np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/X_train.npy')
    y_train = np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/y_train.npy')
    X_val = np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/X_val.npy')
    y_val = np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TrainData/y_val.npy')
    return X_train, y_train, X_val, y_val

#def load_only_testing():
   # X_test1 =  np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TestData/X_test1.npy')
  #  y_test1 = np.load('E:/Users/Atithi/Downloads/Project/venv/Promise12/TestData/y_test1.npy')
   # return X_test1, y_test1

