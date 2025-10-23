import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
from tensorflow.image import resize

class MultiScaleSelfAttention(layers.Layer):
    def __init__(self, num_heads=8, key_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shapes):
        self.Wq = [layers.Dense(self.key_dim) for _ in input_shapes]
        self.Wk = [layers.Dense(self.key_dim) for _ in input_shapes]
        self.Wv = [layers.Dense(self.key_dim) for _ in input_shapes]
        self.dense = [layers.Dense(shape[-1]) for shape in input_shapes]

    def call(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            q = self.Wq[i](input)
            k = self.Wk[i](input)
            v = self.Wv[i](input)
            attention_scores = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32)))
            attention_output = tf.matmul(attention_scores, v)
            outputs.append(self.dense[i](attention_output))
        return outputs

def build_mssa_densenet_model(input_shape=(224, 224, 3), num_classes={'Phylum': 4, 'Class': 3, 'Genus': 10, 'Family': 10, 'Order': 10, 'Species': 10}):
    # Load DenseNet121 as the base model
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Define the layers to use for MSSA
    layer_names = ['pool2_pool', 'pool3_pool', 'pool4_pool']  # Example layers from DenseNet121
    layers_output = [base_model.get_layer(name).output for name in layer_names]
    target_shape = layers_output[0].shape[1:3]  # The shape of the highest resolution output

    # Resize outputs to the same shape
    resized_outputs = [resize(output, target_shape) for output in layers_output]

    # Apply MSSA
    mssa_outputs = MultiScaleSelfAttention(num_heads=8, key_dim=64)(resized_outputs)
    concatenated = layers.Concatenate()(mssa_outputs)  
    x = layers.GlobalAveragePooling2D()(concatenated)

    # Add custom layers on top of the MSSA output
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Create branches for each category in a cascaded manner
    outputs = []
    inputs = x  # Initial input for the cascaded classifiers
    for category in ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']:
        branch = layers.Dense(256, activation='relu')(inputs)
        output = layers.Dense(num_classes[category], activation='softmax', name=f'{category.lower()}_output')(branch)
        outputs.append(output)
        inputs = layers.Concatenate()([inputs, output])  # Cascade the output for the next branch

    # Create the model with cascaded branches
    model = models.Model(inputs=base_model.input, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss={f'{category.lower()}_output': 'sparse_categorical_crossentropy' for category in num_classes},
                  metrics=['accuracy'])
    return model

# Build and summarize the model with DenseNet121 as backbone
model = build_mssa_densenet_model()
model.summary()