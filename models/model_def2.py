import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def conv_block(x, filters, pool=False, l2=1e-4):
    """
    Conv2D → BatchNorm → ReLU (→ optional MaxPool).
    Matches the Colab architecture exactly.
    """
    x = layers.Conv2D(
        filters, (3, 3), 
        padding='same', 
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2),
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if pool:
        x = layers.MaxPooling2D((2, 2))(x)
    return x

def build_wandiya_model_tf(img_size=150, num_classes=6):
    """
    Constructs the IntelCNN architecture using the Keras Functional API.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name='input_image')

    # Stage 1
    x = conv_block(inputs, 32)
    x = conv_block(x, 64,  pool=True)

    # Stage 2
    x = conv_block(x, 128)
    x = conv_block(x, 128, pool=True)

    # Stage 3
    x = conv_block(x, 256)
    x = conv_block(x, 256, pool=True)

    # Stage 4
    x = conv_block(x, 512)
    x = conv_block(x, 512, pool=True)

    # Head / Classifier
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    x = layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        kernel_initializer='glorot_uniform'
    )(x)
    
    x = layers.Dropout(0.4)(x)
    
    # Final Layer with Softmax
    outputs = layers.Dense(
        num_classes, 
        activation='softmax',
        name='predictions'
    )(x)

    return keras.Model(inputs, outputs, name='IntelCNN')