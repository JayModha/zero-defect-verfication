from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.models import Model

def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D()(c2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    # Up
    u1 = UpSampling2D()(c3)
    m1 = concatenate([u1, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(m1)
    u2 = UpSampling2D()(c4)
    m2 = concatenate([u2, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(m2)
    outputs = Conv2D(1, 1, activation='sigmoid')(c5)
    return Model(inputs, outputs)
