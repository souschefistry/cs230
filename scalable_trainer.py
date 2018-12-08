"""
    Copyright (c) 2018 dibghosh AT stanford edu

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software
    is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import sys
print(sys.path)
print(sys.executable)
import os
import numpy as np
import json
import random
import numpy as np
import os
import time
import keras
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import TensorBoard
from keras import optimizers

from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle

from keras.utils.data_utils import get_file
from keras.models import model_from_json
from keras import backend as K

import tensorflow as tf
import functools

from glob import glob

# lscpu Core(s) per socket: 2
NUM_PARALLEL_EXEC_UNITS = 2
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session(
    config=tf.ConfigProto(
        log_device_placement=True,
        intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, 
        inter_op_parallelism_threads=2,
        allow_soft_placement=True,
#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
        device_count = {'GPU': 1, 'CPU': NUM_PARALLEL_EXEC_UNITS }
    )
)
keras.backend.set_session(sess)

NUM_CLASSES = 46
TRAIN_DATA_SIZE = 5000
TEST_DATA_SIZE = 1000
VAL_DATA_SIZE = 1000
img_h = 224
img_w = 224

np.random.seed(seed=1234)
random.seed(1234)

### write both training + validation graphs in same plot
# https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

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

### train + data process settings

NUM_EPOCHS = 1
n_H, n_W = 224, 224
TRAIN_BATCH_SIZE = 128
WARM_UP_EPOCHS = 1
FINE_TUNING_EPOCHS = 1
nb_train_samples = 35979
nb_validation_samples= 19140
nb_test_samples = 40000
ALPHA_LEARNING_RATE = 0.001

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

# no augmentation for test generator
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '../deepfashion/dataset/train/',
        target_size=(n_H, n_W),
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '../deepfashion/dataset/val/',
        target_size=(n_H, n_W),
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        '../deepfashion/dataset/test/',
        target_size=(n_H, n_W),
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='categorical')

base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(n_H, n_W, 3))

base_model.summary()

last_layer = base_model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(1024, activation='relu',name='fc-1')(x)
# x = Dropout(0.5)(x)
# x = Dense(256, activation='relu',name='fc-2')(x)
# x = Dropout(0.5)(x)
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

# first warm up last layer with few initial runs

t=time.time()
with tf.device('/gpu:0'):
    history = custom_resnet_model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples // TRAIN_BATCH_SIZE,
                epochs=WARM_UP_EPOCHS,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples // TRAIN_BATCH_SIZE,
                verbose=1)
print('Training time (secs): %s' % (time.time() - t))

# get accuracy numbers
with tf.device('/gpu:0'):
    (loss, accuracy) = custom_resnet_model.evaluate_generator(
        test_generator,
        steps=nb_test_samples // TRAIN_BATCH_SIZE,
        verbose=1)

print("[INFO] pre fine-tune loss={:.4f}, pre fine-tune accuracy: {:.4f}%".format(loss,accuracy * 100))

# define tensorboard details
tensorboard = TensorBoard(log_dir="./deepfashion/tboard-resnet50-scalable/{}_{}_{}".format(TRAIN_BATCH_SIZE, FINE_TUNING_EPOCHS, time.time()), write_graph=True)

# improvement #1
# early_stopping = EarlyStopping(verbose=1, patience=40, monitor='val_loss')
# callbacks_list = [early_stopping, tensorboard]

# tensorboard = TrainValTensorBoard(log_dir="./deepfashion/tboard-resnet50-logs/{}_{}_{}".format(BATCH_SIZE, NUM_EPOCHS, time.time()), write_graph=True)

callbacks_list = [tensorboard]

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)
    
# Test # 1: we chose to train the top 1 resnet blocks, i.e. we will freeze
# the first 163 layers and unfreeze the rest: (add_31)
# for layer in base_model.layers[:163]:
#     layer.trainable = False
# for layer in base_model.layers[163:]:
#     layer.trainable = True   
    
# Test # 2: we chose to train the top 2 resnet blocks, i.e. we will freeze
# the first 153 layers and unfreeze the rest: (add_30)
# for layer in base_model.layers[:153]:
#     layer.trainable = False
# for layer in base_model.layers[153:]:
#     layer.trainable = True
    
# Test # 3: we chose to train the top 3 resnet blocks, i.e. we will freeze
# the first 143 layers and unfreeze the rest: (add_29)
# for layer in base_model.layers[:141]:
#     layer.trainable = False
# for layer in base_model.layers[141:]:
#     layer.trainable = True 

# Test # 4: we chose to train the top 4 resnet blocks, i.e. we will freeze
# the first 143 layers and unfreeze the rest: (add_28)
for layer in base_model.layers[:131]:
    layer.trainable = False
for layer in base_model.layers[131:]:
    layer.trainable = True     

top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'

opti_grad_clip=optimizers.Adam(lr=ALPHA_LEARNING_RATE)

# opti_grad_clip=optimizers.RMSprop(lr=2e-3)
custom_resnet_model.compile(loss='categorical_crossentropy', optimizer=opti_grad_clip, metrics=['accuracy', 'top_k_categorical_accuracy', top3_acc])

# init plotter
# plot_losses = TrainingPlot(FINE_TUNING_EPOCHS, TRAIN_BATCH_SIZE)
# callbacks_list.append(plot_losses)

lr_decay = LearningRateScheduler(schedule=lambda epoch: ALPHA_LEARNING_RATE * (0.9 ** epoch))
callbacks_list.append(lr_decay)

# run full training with fine tuning
t=time.time()
with tf.device('/gpu:0'):
    history = custom_resnet_model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples // TRAIN_BATCH_SIZE,
                epochs=FINE_TUNING_EPOCHS,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples // TRAIN_BATCH_SIZE,
                verbose=1,
                callbacks=callbacks_list)
print('Training time (secs): %s' % (time.time() - t))

# run accuracy calculation after finetuning
with tf.device('/gpu:0'):
    (loss, accuracy, top_5, top_3) = custom_resnet_model.evaluate_generator(
        test_generator,
        steps=nb_test_samples // TRAIN_BATCH_SIZE,
        verbose=1)

print("[INFO] final loss={:.4f}, final accuracy: {:.4f}, final top_5: {:.4f}, final top_3: {:.4f}%".format(loss, accuracy * 100, top_5, top_3))

# t=time.time()
# with tf.device('/gpu:0'):
#     history = custom_resnet_model.fit_generator(
#                 train_generator,
#                 epochs=NUM_EPOCHS,
#                 validation_data=validation_generator,
#                 verbose=1,
#                 callbacks=callbacks_list)
# print('Training time (secs): %s' % (time.time() - t))

# # store the model on disk
# model_name = 'scalable_resnet50_{}_{}_{}.h5'.format(TRAIN_BATCH_SIZE, EPOCHS, time.time())
# save_model_to_disk(custom_resnet_model, model_name)


# print('STATIC LEARNING_PHASE = 1')
# K.clear_session()
# K.set_learning_phase(1)

# # load saved model
# custom_resnet_model = load_model_from_disk(model_name)
# custom_resnet_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

# with tf.device('/gpu:0'):
#     (loss, accuracy) = custom_resnet_model.evaluate_generator(
#         test_generator,
#         verbose=1)


# store the model on disk
# model_name = 'scalable_batch_resnet50_{}_{}_{}.h5'.format(TRAIN_BATCH_SIZE, EPOCHS, time.time())
# save_model_to_disk(custom_resnet_model, model_name)

# print('STATIC LEARNING_PHASE = 1')
# K.clear_session()
# K.set_learning_phase(1)

# # load saved model
# custom_resnet_model = load_model_from_disk(model_name)
# custom_resnet_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

# visualize results

"""
# Create a generator for prediction
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)
 
# Get the filenames from the generator
fnames = validation_generator.filenames
 
# Get the ground truth from generator
ground_truth = validation_generator.classes
 
# Get the label to class mapping from the generator
label2index = validation_generator.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
 
# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
 
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
 
# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]
     
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])
     
    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()
"""
