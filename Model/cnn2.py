import keras,os
from keras.preprocessing import image
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def get_model(height, width):
    model = Sequential()

    model.add( Conv2D( 128, (3,3), activation="relu", input_shape = (height, width, 3) ) )
    model.add( MaxPool2D( pool_size=(2,2) ) )

    model.add( Conv2D( 64, (3,3), activation="relu" ) )
    model.add( MaxPool2D(pool_size=(2,2)) )

    model.add( Conv2D( 32, (3,3), activation="relu" ) )
    model.add( MaxPool2D(pool_size=(2,2)) )

    model.add( Conv2D( 16, (3,3), activation="relu" ) )
    model.add( MaxPool2D(pool_size=(2,2)) )

    model.add(Flatten())

    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))    
    model.add(Dense(10, activation="softmax"))

    return model





def main():
    height = 128
    width = 128
    train_obj = ImageDataGenerator()
    train_data = train_obj.flow_from_directory(directory="augmented_train", target_size=(width, height))
    test_obj = ImageDataGenerator()
    test_data = test_obj.flow_from_directory(directory="new_test", target_size=(width,height))


    model = get_model(height, width)
    model.compile(optimizer="adam", metrics=['accuracy'], loss=keras.losses.categorical_crossentropy )
    #print(train_data)
    #model.summary()

    check_point = ModelCheckpoint("model_1.h5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early_stop = EarlyStopping(monitor='acc', min_delta=20, verbose=1, mode='auto', patience=5)

    khela = model.fit_generator(steps_per_epoch=100, generator=train_data, validation_data=test_data, validation_steps=10, epochs=100 , callbacks=[check_point, early_stop])
    
    train_loss = khela.history['loss']
    test_loss = khela.history['val_loss']
    train_acc = khela.history['acc']
    test_acc = khela.history['val_acc']
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