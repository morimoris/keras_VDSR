import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, Add

def VDSR(depth): 
    """
    depth : number of residual blocks.
    """
    input_shape = Input((None, None, 1))

    conv2d_1 = Conv2D(filters = 64,
                        kernel_size = (3, 3),
                        padding = "same",
                        activation = "relu"
                        )(input_shape)

    conv2d_residual = conv2d_1
    for i in range(depth - 2):
        conv2d_residual = Conv2D(filters = 64,
                                kernel_size = (3, 3),
                                padding = "same",
                                activation = "relu"
                                )(conv2d_residual)

    conv2d_depth = Conv2D(filters = 1,
                    kernel_size = (3, 3),
                    padding = "same"
                    )(conv2d_residual)

    skip_connection = Add()([input_shape, conv2d_depth])

    model = Model(inputs = input_shape, outputs = skip_connection)

    model.summary()

    return model

model = VDSR
