from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras


def make_model_res_net_50(input_shape):
    res_net_model = keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    model = keras.models.Sequential()
    model.add(res_net_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def make_model_cnn(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.9))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.8))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.75))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.75))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.75))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dropout(0.5))
    model.add(BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
