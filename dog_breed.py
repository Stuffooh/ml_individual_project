import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from datetime import timedelta
import math
import os
import scipy.misc
from scipy.stats import itemfreq
from random import sample
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import PIL.Image
from IPython.display import display
from zipfile import ZipFile
from io import BytesIO

archive_train = ZipFile("/home/kevinvandijk/ML_course/dataset/train.zip", 'r')
archive_test = ZipFile("/home/kevinvandijk/ML_course/dataset/test.zip", 'r')

len(archive_train.namelist()[:]) - 1

def DataBase_creator(archivezip, nwidth, nheight, save_name):
    start_time = time.time()
    
    s = (len(archivezip.namelist()[:])-1, nwidth, nheight,3)
    allImage = np.zeros(s)
    for i in range(1,len(archivezip.namelist()[:])):
        filename = BytesIO(archivezip.read(archivezip.namelist()[i]))
        image = PIL.Image.open(filename) # open colour image
        image = image.resize((nwidth, nheight))
        image = np.array(image)
        image = np.clip(image/255.0, 0.0, 1.0) # 255 = max of the value of a pixel
        allImage[i-1]=image
    
    pickle.dump(allImage, open( save_name + '.p', "wb" ) )
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

image_resize = 60
DataBase_creator(archivezip = archive_train, nwidth = image_resize, nheight = image_resize , save_name = 'train')
DataBase_creator(archivezip = archive_test, nwidth = image_resize, nheight = image_resize , save_name = 'test')

nwidth = image_resize
nheight = image_resize
df_train = pd.read_csv('/home/kevinvandijk/ML_course/dataset/labels.csv')

s = (len(df_train['breed']), nwidth, nheight,3)
allImage = np.zeros(s)

i = 0
for f, breed in df_train.values:
    image = PIL.Image.open('/home/kevinvandijk/ML_course/dataset/train/{}.jpg'.format(f))
    image = image.resize((nwidth, nheight))
    image = np.array(image)
    image = np.clip(image/255.0, 0.0, 1.0) # 255 = max of the value of a pixel
    i += 1
    allImage[i-1]=image

train = allImage

labels_raw = pd.read_csv("/home/kevinvandijk/ML_course/dataset/labels.csv", header=0, sep=',', quotechar='"')


Nber_of_breeds = 120

def main_breeds(labels_raw, Nber_breeds , all_breeds='TRUE'):
    labels_freq_pd = itemfreq(labels_raw["breed"])
    labels_freq_pd = labels_freq_pd[labels_freq_pd[:, 1].argsort()[::-1]]
    
    if all_breeds == 'FALSE':
        main_labels = labels_freq_pd[:,0][0:Nber_breeds]
    else: 
        main_labels = labels_freq_pd[:,0][:]
        
    labels_raw_np = labels_raw["breed"].as_matrix()
    labels_raw_np = labels_raw_np.reshape(labels_raw_np.shape[0],1)
    labels_filtered_index = np.where(labels_raw_np == main_labels)
    return labels_filtered_index

labels_filtered_index = main_breeds(labels_raw = labels_raw, Nber_breeds = Nber_of_breeds, all_breeds='FALSE')
labels_filtered = labels_raw.iloc[labels_filtered_index[0],:]
train_filtered = train[labels_filtered_index[0],:,:,:]
print('- Number of images remaining after selecting the {0} main breeds : {1}'.format(Nber_of_breeds, labels_filtered_index[0].shape))
print('- The shape of train_filtered dataset is : {0}'.format(train_filtered.shape))

labels = labels_filtered["breed"].as_matrix()
labels = labels.reshape(labels.shape[0],1) #labels.shape[0] looks faster than using len(labels)

def matrix_Bin(labels):
    labels_bin=np.array([])
    labels_name, labels0 = np.unique(labels, return_inverse=True)
    labels0
    for _, i in enumerate(itemfreq(labels0)[:,0].astype(int)):
        labels_bin0 = np.where(labels0 == itemfreq(labels0)[:,0][i], 1., 0.)
        labels_bin0 = labels_bin0.reshape(1,labels_bin0.shape[0])
        if (labels_bin.shape[0] == 0):
            labels_bin = labels_bin0
        else:
            labels_bin = np.concatenate((labels_bin,labels_bin0 ),axis=0)
    print("Nber SubVariables {0}".format(itemfreq(labels0)[:,0].shape[0]))
    labels_bin = labels_bin.transpose()
    print("Shape : {0}".format(labels_bin.shape))
    return labels_name, labels_bin

labels_name, labels_bin = matrix_Bin(labels = labels)

for breed in range(len(labels_name)):
    print('Breed {0} : {1}'.format(breed,labels_name[breed]))

labels_cls = np.argmax(labels_bin, axis=1)

num_validation = 0.30

X_train, X_validation, y_train, y_validation = train_test_split(train_filtered, labels_bin, test_size=num_validation, random_state=6)

def train_test_creation(x, data, toPred): 
    indices = sample(range(data.shape[0]),int(x * data.shape[0])) 
    indices = np.sort(indices, axis=None) 
  
    index = np.arange(data.shape[0]) 
    reverse_index = np.delete(index, indices,0)
  
    train_toUse = data[indices]
    train_toPred = toPred[indices]
    test_toUse = data[reverse_index]
    test_toPred = toPred[reverse_index]
    return train_toUse, train_toPred, test_toUse, test_toPred

df_train_toUse, df_train_toPred, df_test_toUse, df_test_toPred = train_test_creation(0.7, train_filtered, labels_bin)

df_validation_toPred_cls = np.argmax(y_validation, axis=1)
df_validation_toPred_cls[0:9]

img_size = image_resize
num_channels = 3
img_size_flat = img_size * img_size
img_shape = (img_size, img_size, num_channels)
num_classes = Nber_of_breeds

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.02))

def new_biases(length):
    return tf.Variable(tf.constant(0.02, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True, use_dropout=True):  
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    layer = tf.nn.relu(layer)
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    if use_dropout:
        layer = tf.nn.dropout(layer,keep_prob_conv)
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True, use_dropout=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    if use_dropout:
        layer = tf.nn.dropout(layer,keep_prob_fc)
    return layer

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
keep_prob_fc=tf.placeholder(tf.float32)
keep_prob_conv=tf.placeholder(tf.float32)

filter_size1 = 5
num_filters1 = 32

filter_size2 = 4        
num_filters2 = 64

filter_size3 = 3
num_filters3 = 128

fc_size = 512

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True,
                   use_dropout=False)
    
layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True,
                   use_dropout=False)
    
layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True,
                   use_dropout=True)

layer_flat, num_features = flatten_layer(layer_conv3)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=False,
                         use_dropout=False)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False,
                         use_dropout=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
def init_variables():
    session.run(tf.global_variables_initializer())

batch_size = 50

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def optimize(num_iterations, X):
    global total_iterations
    start_time = time.time()
    losses = {'train':[], 'validation':[]}
    for i in range(num_iterations):
        total_iterations += 1
        x_batch, y_true_batch = next_batch(batch_size, X_train, y_train)
        feed_dict_train = {x: x_batch, y_true: y_true_batch, keep_prob_conv : 0.3,keep_prob_fc : 0.4}
        feed_dict_validation = {x: X_validation, y_true: y_validation, keep_prob_conv : 1, keep_prob_fc : 1}
        session.run(optimizer, feed_dict=feed_dict_train)
        acc_train = session.run(accuracy, feed_dict=feed_dict_train)
        acc_validation = session.run(accuracy, feed_dict=feed_dict_validation)
        losses['train'].append(acc_train)
        losses['validation'].append(acc_validation)
        if (total_iterations % X == 0) or (i ==(num_iterations -1)):
            msg = "Iteration: {0:>6}, Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}"
            print(msg.format(total_iterations, acc_train, acc_validation))
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    _ = plt.ylim()

init_variables()
total_iterations = 0
optimize(num_iterations=10000, X=250) 
