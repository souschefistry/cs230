Copyright © 2018 dghosh

import sys
print(sys.path)
print(sys.executable)
import os
import numpy as np
import json
import random

import keras
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.models import model_from_json

import tensorflow as tf

# lscpu Core(s) per socket: 2
NUM_PARALLEL_EXEC_UNITS = 2
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'GPU': 1, 'CPU': NUM_PARALLEL_EXEC_UNITS })
keras.backend.set_session(sess)

NUM_CLASSES = 46
TRAIN_DATA_SIZE = 5000
TEST_DATA_SIZE = 1000
VAL_DATA_SIZE = 1000
img_h = 224
img_w = 224

np.random.seed(seed=1234)
random.seed(1234)

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x

from glob import glob
from keras.preprocessing import image
from tqdm import tqdm_notebook, tqdm # Iteration visualization

def load_dataset(data_dir_list, max_per_class=100):
    """
        loads images in memory. Expensive method. Doesn't scale well
    """
    
    img_data_list, labels =[],[]

    for dataset in tqdm(data_dir_list):
        img_list=glob("../deepfashion/dataset/train/%s/*.jpg" % dataset)
        print ('Loaded {} images out of {} for category {}'.format(len(img_list[:max_per_class]), len(img_list), dataset))
        for img_path in img_list[:max_per_class]:
            labels.append(dataset)
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            img_data_list.append(x)

    img_data = np.array(img_data_list)
    img_data=np.rollaxis(img_data,1,0)
    img_data=img_data[0]
    return img_data, labels

def save_model_to_disk(model, model_name="model"):
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s.json" % model_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("%s.h5" % model_name)
    print("Saved %s to disk" % model_name)
    
def load_model_from_disk(model_name):
    # load json and create model
    json_file = open('%s.json' % model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s.h5" % model_name)
    print("Loaded %s from disk" % model_name)
    return loaded_model

train_data_dir = os.listdir("../deepfashion/dataset/train/")
val_data_dir = os.listdir("../deepfashion/dataset/val/")
test_data_dir = os.listdir("../deepfashion/dataset/test/")

images_per_class = 200
print("[*] preparing train data with max %s images per class" % images_per_class)
train_data, train_labels = load_dataset(train_data_dir, images_per_class)
print("[*] preparing val data with max %s images per class" % images_per_class)
val_data, val_labels = load_dataset(val_data_dir, images_per_class)
print("[*] preparing test data with max %s images per class" % images_per_class)
test_data, test_labels = load_dataset(test_data_dir, images_per_class)

# train_set = get_dataset(train_data_dir)
# val_set = get_dataset(val_data_dir)

# nrof_classes = len(train_set)

# print('Number of classes : %s' % nrof_classes)

# # prepare data
# # convert class labels to on-hot encoding
# # Get a list of image paths and their labels
# train_image_list, train_label_list = get_image_paths_and_labels(train_set)
# assert len(image_list) > 0, 'The training set should not be empty'
# val_image_list, val_label_list = get_image_paths_and_labels(val_set)

# we will use the encoders from the scikit-learn library. 
# Specifically, the LabelEncoder of creating an integer encoding of labels 
# and the OneHotEncoder for creating a one hot encoding of integer encoded values.
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train_labels)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
train_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(val_labels)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
val_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(test_labels)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
test_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

### network

import numpy as np
import os
import time
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.applications import ResNet50
from keras.callbacks import TensorBoard
from keras import optimizers

from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
# from sklearn.cross_validation import train_test_split
# keras BN bug - https://github.com/keras-team/keras/pull/9965
K.clear_session()
K.set_learning_phase(0)

base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(img_h, img_w, 3))

base_model.summary()

last_layer = base_model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(1024, activation='relu',name='fc-1')(x)
# a softmax layer for 4 classes
predictions = Dense(NUM_CLASSES, activation='softmax',name='output_layer')(x)

# this is the model we will train
custom_resnet_model = Model(inputs=base_model.input, outputs=predictions)

custom_resnet_model.summary()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in custom_resnet_model.layers:
    layer.trainable = False

# custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

# train settings
TRAIN_BATCH_SIZE = 128
EPOCHS = 100

tensorboard = TensorBoard(log_dir="./deepfashion/tboard-resnet50-logs/{}_{}_{}".format(TRAIN_BATCH_SIZE, EPOCHS, time.time()), write_graph=True)

t=time.time()
with tf.device('/gpu:0'):
    hist = custom_resnet_model.fit(
        train_data, 
        train_onehot_encoded, 
        batch_size=TRAIN_BATCH_SIZE, 
        epochs=EPOCHS, 
        verbose=1, 
        validation_data=(val_data, val_onehot_encoded),
        callbacks=[tensorboard])
print('Training time (secs): %s' % (time.time() - t))

# Store the model on disk
model_name = 'resnet50_{}_{}_{}.h5'.format(TRAIN_BATCH_SIZE, EPOCHS, time.time())
save_model_to_disk(custom_resnet_model, model_name)

print('STATIC LEARNING_PHASE = 1')
K.clear_session()
K.set_learning_phase(1)
custom_resnet_model = load_model_from_disk(model_name)
custom_resnet_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

with tf.device('/gpu:0'):
    (loss, accuracy) = custom_resnet_model.evaluate(test_data, 
                                                     test_onehot_encoded, 
                                                     batch_size=TRAIN_BATCH_SIZE, 
                                                     verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name)

