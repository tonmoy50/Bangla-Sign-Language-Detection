import keras,os
from keras import applications
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np



def get_model(height, width):
    conv_1 = 32
    conv_2 = 64
    conv_3 = 128
    conv_4 = 256
    conv_5 = 512

    model = Sequential()

    model.add( Conv2D( input_shape=(height, width, 3), filters=conv_1, kernel_size=(7,7), strides = (2, 2), padding="same", activation="relu" ) )
    
    model.add( MaxPool2D(pool_size=(3,3), strides=(2,2) ) )

    resnet = 10



    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model


def main():
    height, width = 224, 224

    train_obj = ImageDataGenerator()
    train_data = train_obj.flow_from_directory(directory="train2", target_size=(width, height))
    test_obj = ImageDataGenerator()
    test_data = test_obj.flow_from_directory(directory="validate", target_size=(width,height))

    #model = get_model(height, width)
    base_model = applications.resnet50.ResNet50(include_top=None, weights=None, input_shape=(height, width, 3) )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(10, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    #model.summary()

    check_point = ModelCheckpoint("model_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early_stop = EarlyStopping(monitor='acc', min_delta=20, verbose=1, mode='auto', patience=5)

    khela = model.fit_generator(steps_per_epoch=10, generator=train_data, validation_data=test_data, validation_steps=10, epochs=100 , callbacks=[check_point, early_stop])


if __name__ == "__main__":
    main()