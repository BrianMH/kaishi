#!/env/bin/python
"""
COGS 181 Project
Author: Brian Henriquez, Simon Fong, Wilson Tran
"""

from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import numpy as np
import os
import cv2
import random
import matplotlib
from datetime import datetime                   # Use to record time
import json                                     # Writing data to logger
matplotlib.use('Agg')                           # Stops from plotting to screen
import matplotlib.pyplot as plt
from dataset import Dataset                     # Custom Dataset

DATASET_NAME = 'plankton'
IMAGE_WIDTH,IMAGE_HEIGHT,NUM_CHANNELS = 299,299,3
EPOCHS = 10
BATCH_SIZE = 50
NUM_TRAIN,NUM_VAL,NUM_TEST = 10,5,85


ID = "{}_{}_{}_{}_{}_{}_{}".format("5De256relue",DATASET_NAME,
                                EPOCHS,BATCH_SIZE,NUM_TRAIN,NUM_VAL,NUM_TEST)


# Load dataset
cal = Dataset(DATASET_NAME,IMAGE_HEIGHT,IMAGE_WIDTH)
cal.read_data()
cal.train_val_test_split(NUM_TRAIN,NUM_VAL,NUM_TEST)
num_classes = cal.num_classes

def load_model():
    """Returns a pretrained model"""
    
    # Loads base model
    base_model = InceptionV3(include_top=False, weights='imagenet',input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
    print("Model weights loaded.")
    
    base_out = base_model.output

    # Add layers more layers
    x = Flatten()(base_out)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256,activation='relu')(x)
    x = Dense(256,activation='relu')(x)
    x = Dense(256,activation='relu')(x)
    x = Dense(256,activation='relu')(x)
    x = Dense(256,activation='relu')(x)

    # Final fully connected layer to work with our data
    predictions = Dense(num_classes,activation='softmax')(x)

    # Build a final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print("Model structure")
    model.summary()
    
    # Compile model
    model.compile(optimizers.SGD(lr=1e-4,momentum=0.9),
                'categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled")

    return model
    
def logger(message):
    """Logs any message into a file"""
    with open('./models/stats.txt', 'a+') as f:
        print >>f, message


def main():
    
    # Make model
    model = load_model()
    print("Model created\n")

    # Init data array to plot
    train_acc = np.array([])
    train_val_acc = np.array([])
    train_loss = np.array([])
    train_val_loss = np.array([])
    
    # Load the training data
    X_train, Y_train = cal.load_training()
    
    # Load the validation data
    X_val, Y_val = cal.load_validation()
    
    # Start time
    start_time = datetime.now()
    print('Start Time', start_time)
    
    # Train model and store stats in history
    history = model.fit(x=X_train,y=Y_train,batch_size=BATCH_SIZE,
                        epochs=EPOCHS,validation_data=(X_val,Y_val))
    
    # End time
    stop_time = datetime.now()
    print('Stop Time', stop_time)
    
    # Print total time
    elapsed_time = stop_time - start_time
    print('My Elapsed Time', elapsed_time)
    logger(elapsed_time)   

    # Append the accuracy and loss scores
    train_acc = np.append(train_acc, history.history['acc'])
    train_val_acc = np.append(train_val_acc, history.history['val_acc'])
    train_loss = np.append(train_loss, history.history['loss'])
    train_val_loss = np.append(train_val_loss, history.history['val_loss'])
    
     
    # Save model weights
    model.save('./models/{}.h5'.format(ID))
    logger(ID)
    logger(history.history)
    print("Model weights saved.")

    
    # Plot accuracy
    plt.hold(True)
    plt.plot(train_acc)
    plt.plot(train_val_acc)
    plt.legend(loc='upper right')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('./plots/acc_vs_val_acc_{}.png'.format(ID))
    plt.hold(False)
    plt.show()

    # Plot loss
    plt.hold(True)
    plt.plot(train_loss)
    plt.plot(train_val_loss)
    plt.legend(loc='upper right')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./plots/loss_vs_val_loss_{}.png'.format(ID))
    plt.hold(False)
    plt.show()
    
    # Test the model
    X_test, Y_test = cal.load_testing()
    metrics = model.evaluate(x=X_test,y=Y_test, batch_size=BATCH_SIZE)
    
    print(metrics)
    logger(metrics)
    print(model.metrics_names)
    logger(model.metrics_names)

    return 0


if __name__ == '__main__':
    main()
