import keras
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add

from keras.regularizers import l2
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt



class ResNet:

    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        shortcut = data
  
        #first branch
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D( int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg) )(act1)     

        #second branch
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D( int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg) )(act2) 

        #third branch
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D( K, (1, 1), use_bias=False, kernel_regularizer=l2(reg) )(act3)        


        if red:
            shortcut = Conv2D( K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg) )(act1)

        
        x = add([conv3, shortcut])

        
        return x             

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        
        inputShape = (height, width, depth)
        chanDim = -1
        

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

            
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)

        x = Conv2D(filters[0], (5, 5), use_bias=False,
        padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        

        for i in range(0, len(stages)):

            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            for j in range(0, stages[i] - 1):
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)


        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name="resnet")

        return model




def main():
    

    width, height = 128, 128
    depth = 3
    classes = 10
    stages = [3,4,6]
    filters = [16,32,64,128]

    train_obj = ImageDataGenerator()
    train_data = train_obj.flow_from_directory(directory="train2", target_size=(width, height))
    test_obj = ImageDataGenerator()
    test_data = test_obj.flow_from_directory(directory="test2", target_size=(width,height))

    check_point = ModelCheckpoint("test_model1.h5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early_stop = EarlyStopping(monitor='acc', min_delta=20, verbose=1, mode='auto', patience=10)


    res_model = ResNet().build(width=width, height=height, depth=depth, classes=classes, stages=stages, filters=filters)
    res_model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    res_model.summary()

    khela = res_model.fit_generator(steps_per_epoch=100, generator=train_data, validation_data=test_data, validation_steps=10, epochs=100 , callbacks=[check_point, early_stop])


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