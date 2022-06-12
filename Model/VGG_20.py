
import keras,os
from keras.preprocessing import image
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



def get_model(height, weight):
    #vgg16_19 model of Convolutional network
    model = Sequential()

    layer1 = 10
    layer2 = 20
    layer3 = 30
    layer4 = 40
    layer5 = 40

    model.add( Conv2D( input_shape=(height, weight, 3), filters=layer1, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.2) )
    model.add( Conv2D( filters=layer1, kernel_size=(3,3), padding="same", activation="relu" ) )

    model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )

    model.add( Conv2D( filters=layer2, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.2) )
    model.add( Conv2D( filters=layer2, kernel_size=(3,3), padding="same", activation="relu" ) )

    model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )

    model.add( Conv2D( filters=layer3, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.2) )
    model.add( Conv2D( filters=layer3, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.2) )
    model.add( Conv2D( filters=layer3, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.2) )
    model.add( Conv2D( filters=layer3, kernel_size=(3,3), padding="same", activation="relu" ) )

    model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )

    model.add( Conv2D( filters=layer4, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.2) )
    model.add( Conv2D( filters=layer4, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.2) )
    model.add( Conv2D( filters=layer4, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.2) )
    model.add( Conv2D( filters=layer4, kernel_size=(3,3), padding="same", activation="relu" ) )

    model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )

    model.add( Conv2D( filters=layer5, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.2) )
    model.add( Conv2D( filters=layer5, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.2) )
    model.add( Conv2D( filters=layer5, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.2) )
    model.add( Conv2D( filters=layer5, kernel_size=(3,3), padding="same", activation="relu" ) )

    model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )
    model.add( BatchNormalization() )

    model.add(Flatten())

    model.add( Dense(200, activation="relu") )
    #model.add( Dropout(0.2) )
    model.add( Dense(200, activation="relu") )

    model.add(Dense(100, activation="relu"))

    model.add( Dense(2, activation="softmax") )

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return model
    




def main():
    height = 224
    width = 224
    train_obj = ImageDataGenerator()
    train_data = train_obj.flow_from_directory(directory="real_train", target_size=(width, height))
    test_obj = ImageDataGenerator()
    test_data = test_obj.flow_from_directory(directory="real_test", target_size=(width,height))

    #x_data, y_data = train_test_split(train_data, test_size=0.2, random_state=1)

    model = get_model(height, width)

    model.summary()



    check_point = ModelCheckpoint("model_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early_stop = EarlyStopping(monitor='acc', min_delta=20, verbose=1, mode='auto', patience=10)

    khela = model.fit_generator(steps_per_epoch=100, generator=train_data, validation_data=test_data, validation_steps=10, epochs=100, callbacks=[check_point, early_stop] )

    
    train_loss = khela.history['acc']
    test_loss = khela.history['val_acc']
    high = early_stop.stopped_epoch + 2
    epoch = range(1, high)

    plt.plot(epoch, train_loss, 'r', label='Training Loss')
    plt.plot(epoch, test_loss, 'b', label='Test Loss')
    plt.title("Loss Curve of Train and Test")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print("Done!!!!!!!!!!!!!!!!!!!!!")



if __name__ == "__main__":
    main()
