import tensorflow as tf

from CNNFeature import CombinedCNNs

def model(dropout=True, is_train=True, num_res_blocks=1, batch_size=256, height=252, width=151, n_class=2, num_channels=3):
    learn_rate = 5e-5 
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
    optim = tf.keras.optimizers.Adam(learning_rate=learn_rate, clipnorm=5)

    cnn_phase1 = CombinedCNNs(dropout, is_train, num_res_blocks, batch_size, height, width, n_class, num_channels)
    model = cnn_phase1.build_model()

    model.compile(loss = loss,
                  optimizer = optim,
                  metrics=['accuracy'])

    return model