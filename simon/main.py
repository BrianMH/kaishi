from inception_v3 import inception_v3
from data import Dataset
import tensorflow as tf
import cv2
import numpy as np

# Constants
IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
EPOCHS = 10
BATCH_SIZE = 10

# Load dataset
ndsb = Dataset('train',IMAGE_HEIGHT,IMAGE_WIDTH)
ndsb.read_data()
num_classes = ndsb.num_classes

# Placeholder inputs and output
inputs = tf.placeholder(tf.float32, [None,IMAGE_HEIGHT,IMAGE_WIDTH,3])
predict = tf.placeholder(tf.float32, [None,1,1,num_classes])

# Get model
y_conv, some = inception_v3(inputs, num_classes=num_classes)

# Init tf variables
init = tf.global_variables_initializer()

# Cross entropy graph
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=predict, logits=y_conv))

# Training graph
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Predictions graph
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(predict, 1))

# Accuracy graph
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(EPOCHS):
        images, labels = ndsb.next_batch(BATCH_SIZE)
        '''
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                inputs: images, predict: labels})
            print('step %d, training accuracy %g' % (i, train_accuracy))'''
        
        _, loss = sess.run([train_step, cross_entropy], 
                            feed_dict={inputs: images, predict: labels})
        

