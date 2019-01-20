from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense, Dropout, ELU, Lambda
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from config import *
from load_data import generate_data_batch, split_train_val


def get_nvidia_model(summary=True):
    """
    Get the keras Model corresponding to the NVIDIA architecture described in:
    Bojarski, Mariusz, et al. "End to end learning for self-driving cars."

    The paper describes the network architecture but doesn't go into details for some aspects.
    Input normalization, as well as ELU activations are just my personal implementation choice.

    :param summary: show model summary
    :return: keras Model of NVIDIA architecture
    """
    init = 'glorot_uniform'

    if K.backend() == 'theano':
        input_frame = Input(shape=(CONFIG['input_channels'], NVIDIA_H, NVIDIA_W))
    else:
        input_frame = Input(shape=(NVIDIA_H, NVIDIA_W, CONFIG['input_channels']))

    # standardize input
    x = Lambda(lambda z: z / 127.5 - 1.)(input_frame)

    x = Convolution2D(24, (5, 5), padding='valid', strides=(2, 2), kernel_initializer=init)(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(36, (5, 5), padding='valid', strides=(2, 2), kernel_initializer=init)(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(48, (5, 5), padding='valid', strides=(2, 2), kernel_initializer=init)(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, (3, 3), padding='valid', kernel_initializer=init)(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, (3, 3), padding='valid', kernel_initializer=init)(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)

    x = Dense(100, kernel_initializer=init)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(50, kernel_initializer=init)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(10, kernel_initializer=init)(x)
    x = ELU()(x)
    out = Dense(1, kernel_initializer=init)(x)

    #model = Model(input=input_frame, output=out)
    model = Model(outputs=out,inputs=input_frame)

    if summary:
        model.summary()

    return model


if __name__ == '__main__':

    # split udacity csv data into training and validation
    train_data, val_data = split_train_val(csv_driving_data='data/driving_log.csv')

    # get network model and compile it (default Adam opt)
    nvidia_net = get_nvidia_model(summary=True)
    nvidia_net.compile(optimizer='adam', loss='mse')

    # json dump of model architecture
    with open('model.json', 'w') as f:
        f.write(nvidia_net.to_json())

    # define callbacks to save history and weights
    checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.h5')

    nvidia_net.fit_generator(
                        steps_per_epoch=int(len(train_data)/CONFIG['batchsize']), # As advised in keras fit_generator() documentation
                        validation_steps=int(len(val_data)/CONFIG['batchsize']),  # As advised in keras fit_generator() documentation
                        generator=generate_data_batch(train_data, augment_data=True, bias=CONFIG['bias']), 
                        callbacks=[checkpointer], 
                        validation_data=generate_data_batch(val_data, augment_data=False, bias=1.0), 
                        epochs=50,
                        use_multiprocessing = True,
                        workers = 2)
