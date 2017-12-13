#!/env/bin/python
"""
COGS 181
"""
# Import native modules
import os
import json
import random
from datetime import datetime                    # Use to record time

# Import Keras functions
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical

# Import matrix and plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')                           # Stops from plotting to screen
import matplotlib.pyplot as plt

# Import OpenCV
import cv2

# Import custom dataset class
from dataset import Dataset                     

DATASET_NAME = 'plankton_resized'
EPOCHS = 1
BATCH_SIZE = 50
NUM_TRAIN,NUM_VAL,NUM_TEST = 1,1,1

IMAGE_WIDTH,IMAGE_HEIGHT,NUM_CHANNELS = 299,299,3


ID = "{}_{}_{}_{}_{}_{}_{}".format("normal_resized",DATASET_NAME,
                                EPOCHS,BATCH_SIZE,NUM_TRAIN,NUM_VAL,NUM_TEST)


# Load dataset
cal = Dataset(DATASET_NAME,IMAGE_HEIGHT,IMAGE_WIDTH,resized=True)
cal.read_data()
cal.train_val_test_split(NUM_TRAIN,NUM_VAL,NUM_TEST)
num_classes = cal.num_classes

def logger(message):
    """Logs any message into a file"""
    with open('./models/stats.txt', 'a+') as f:
        print >>f, message
        print(message)

def plot(datas, title, xlabel, ylabel, file_name):
    """Plots the data"""
    plt.figure()
    for key,value in datas.iteritems():
        plt.plot(value, label=key)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plots_dir = 'plots'
    file_name + '.png'
    plot_path = os.path.join(plots_dir,file_name)    
    plt.savefig(plot_path)

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

def main():
    
    description = "Base Inception V3"
    
    logger(ID)
    logger(description)
    
    # Make model
    model = load_model()
    print("Model created")

    # Start time
    start_time = datetime.now()

    # Load the training data
    X_train, Y_train = cal.load_training()
    
    # Print time loading training
    elapsed_time = datetime.now() - start_time
    logger("Elapsed Time: Loading training")
    logger(elapsed_time)   
    
    # Start time
    start_time = datetime.now()
    
    # Load the validation data
    X_val, Y_val = cal.load_validation()
    
    # Print time loading validation
    elapsed_time = datetime.now() - start_time
    logger("Elapsed Time: Loading validation")
    logger(elapsed_time)  
    
    # Start time
    start_time = datetime.now()
    
    # Train model and store stats in history
    history = model.fit(x=X_train,y=Y_train,batch_size=BATCH_SIZE,
                        epochs=EPOCHS,validation_data=(X_val,Y_val))
                        
    history = history.history
    
    # Print total time
    elapsed_time = datetime.now() - start_time
    logger("Elapsed Time: Training")
    logger(elapsed_time)   
     
    # Save model weights
    model_dir = 'models'
    
    model_name = '{}'.format(ID)
    model_name += '.h5'
    model_path = os.path.join(model_dir,model_name)
    
    model.save(model_path)

    logger(history)
    print("Model weights saved")
    
    accuracy = {'Training': history['acc'],
                'Validation': history['val_acc']}
                
    loss = {'Training': history['loss'],
            'Validation': history['val_loss']}
    
    # Plot training vs validation accuracy  
    plot(accuracy, "Accuracy: Training vs Validation",
            'Epochs', 'Accuracy', '{}_accuracy_train_val'.format(ID))
    
    # Plot training vs validation loss  
    plot(loss, "Loss: Training vs Validation",
            'Epochs', 'Loss', '{}_loss_train_val'.format(ID))
    
    # Start time
    start_time = datetime.now()
    
    # Loading testing data
    X_test, Y_test = cal.load_testing()
    
    # Print time loading testing
    elapsed_time = datetime.now() - start_time
    logger("Elapsed Time: Loading testing")
    logger(elapsed_time)  
    
    # Start time
    start_time = datetime.now()
    
    # Test model
    metrics = model.evaluate(x=X_test,y=Y_test, batch_size=BATCH_SIZE)
    
    # Print time testing
    elapsed_time = datetime.now() - start_time
    logger("Elapsed Time: Testing")
    logger(elapsed_time)  
    
    logger(metrics)
    logger(model.metrics_names)


if __name__ == '__main__':
    main()
