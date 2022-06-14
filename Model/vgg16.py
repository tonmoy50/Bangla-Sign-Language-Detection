import keras,os
from keras.preprocessing import image
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Model():


    def get_model(self, height, weight):
        #vgg16_19 model of Convolutional network
        model = Sequential()

        layer1 = 8
        layer2 = 16
        layer3 = 32
        layer4 = 64
        layer5 = 64

        model.add( Conv2D( input_shape=(height, weight, 3), filters=layer1, kernel_size=(3,3), padding="same", activation="relu" ) )
        #model.add( Dropout(0.5) )
        model.add( Conv2D( filters=layer1, kernel_size=(3,3), padding="same", activation="relu" ) )

        model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )
        #model.add( BatchNormalization() )

        model.add( Conv2D( filters=layer2, kernel_size=(3,3), padding="same", activation="relu" ) )
        #model.add( Dropout(0.5) )
        model.add( Conv2D( filters=layer2, kernel_size=(3,3), padding="same", activation="relu" ) )

        model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )
        #model.add( BatchNormalization() )

        model.add( Conv2D( filters=layer3, kernel_size=(3,3), padding="same", activation="relu" ) )
        #model.add( Dropout(0.5) )
        model.add( Conv2D( filters=layer3, kernel_size=(3,3), padding="same", activation="relu" ) )
        #model.add( Dropout(0.5) )
        model.add( Conv2D( filters=layer3, kernel_size=(3,3), padding="same", activation="relu" ) )
        

        model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )
        #model.add( BatchNormalization() )

        model.add( Conv2D( filters=layer4, kernel_size=(3,3), padding="same", activation="relu" ) )
        #model.add( Dropout(0.5) )
        model.add( Conv2D( filters=layer4, kernel_size=(3,3), padding="same", activation="relu" ) )
        #model.add( Dropout(0.5) )
        model.add( Conv2D( filters=layer4, kernel_size=(3,3), padding="same", activation="relu" ) )
        

        model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )
        #model.add( BatchNormalization() )

        model.add( Conv2D( filters=layer5, kernel_size=(3,3), padding="same", activation="relu" ) )
        #model.add( Dropout(0.5) )
        model.add( Conv2D( filters=layer5, kernel_size=(3,3), padding="same", activation="relu" ) )
        #model.add( Dropout(0.5) )
        model.add( Conv2D( filters=layer5, kernel_size=(3,3), padding="same", activation="relu" ) )
        

        model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )
        model.add( BatchNormalization() )

        model.add(Flatten())
        

        model.add( Dense(1024, activation="relu") ) 
        #model.add( Dropout(0.5) )
        model.add( Dense(1024, activation="relu") )

        

        model.add( Dense(7, activation="softmax") )

        model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

        return model
    

def mains():
    model_obj = Model()
    height = 128
    width = 128

    model = model_obj.get_model(height, width)

    #print(train_data)
    model.summary()


def main():
    model_obj = Model()
    height = 128
    width = 128
    train_obj = ImageDataGenerator()
    train_data = train_obj.flow_from_directory(directory="BSL AUG", target_size=(width, height))
    test_obj = ImageDataGenerator()
    test_data = test_obj.flow_from_directory(directory="BSL AUG Test", target_size=(width,height))

    #x_data, y_data = train_test_split(train_data, test_size=0.2, random_state=1)

    model = model_obj.get_model(height, width)

    #print(train_data)
    model.summary()

    #model.add( Dense(30, activation="relu") )
    #model.add( Dense(20, activation="relu") )
    #model.add( Dense(10, activation="softmax") )

    check_point = ModelCheckpoint("test_model1.h5", monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early_stop = EarlyStopping(monitor='accuracy', min_delta=20, verbose=1, mode='auto', patience=10)

    khela = model.fit_generator(steps_per_epoch=100, generator=train_data, validation_data=test_data, validation_steps=10, epochs=100 , callbacks=[check_point, early_stop])

    print(khela.history)

    train_loss = khela.history['loss']
    test_loss = khela.history['val_loss']
    train_acc = khela.history['accuracy']
    test_acc = khela.history['val_accuracy']
    high = early_stop.stopped_epoch + 2
    epoch = range(1, high)

    plt.plot(epoch, train_loss, 'r', label='Training loss')
    plt.plot(epoch, test_loss, 'b', label='Test loss')
    plt.title("loss Curve of Train and Test")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(epoch, train_acc, 'r', label='Training Accuracy')
    plt.plot(epoch, test_acc, 'b', label='Test Accuracy')
    plt.title("Accuracy Curve of Train and Test")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    print("Done!!!!!!!!!!!!!!!!!!!!!")



if __name__ == "__main__":
    main()