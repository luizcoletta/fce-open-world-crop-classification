import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VAE:

    def __init__(self,x_train, y_train, x_test, y_test,epoch,lat_dim=2,shape=(28,28,1),len_train=60000):
        #len_train é o numero de objetos no conjunto de treinamento --> 60000 para MNIST
        #shape --> formato dos dados do dataset

        self.encoder, self.decoder = self.VAE_structure(lat_dim,shape)
        self.output = self.VAE_Model(self.encoder, self.decoder,x_train, y_train, x_test, y_test,epoch,len_train)



    def VAE_structure(self,lat_dim,shape):
        # ENCODER
        class Sampling(layers.Layer):
            """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        latent_dim = lat_dim

        encoder_inputs = keras.Input(shape=shape)
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        # encoder.summary()

        # DECODER
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        # decoder.summary()

        return encoder, decoder

    def VAE_Model(self, encoder, decoder,x_train, y_train, x_test, y_test,epoch, len_train):
        # JUNTANDO ENCODER-DECODER
        class modelo(keras.Model):
            def __init__(self, encoder, decoder, **kwargs):
                super(modelo, self).__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder

            def train_step(self, data):
                if isinstance(data, tuple):
                    data = data[0]
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = encoder(data)
                    reconstruction = decoder(z)
                    reconstruction_loss = tf.reduce_mean(
                        keras.losses.binary_crossentropy(data, reconstruction)
                    )
                    reconstruction_loss *= 28 * 28
                    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                    kl_loss = tf.reduce_mean(kl_loss)
                    kl_loss *= -0.5
                    total_loss = reconstruction_loss + kl_loss
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                return {
                    "loss": total_loss,
                    "reconstruction_loss": reconstruction_loss,
                    "kl_loss": kl_loss,
                }


        #(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        dataset_features = np.concatenate([x_train, x_test], axis=0)
        dataset_features = np.expand_dims(dataset_features, -1).astype("float32") / 255

        vae = modelo(encoder, decoder)

        vae.compile(optimizer=keras.optimizers.Adam())
        vae.fit(dataset_features, epochs=epoch, batch_size=128)
        z_mean, z_log_var, z = vae.encoder(dataset_features)

        latent_features = np.concatenate([z_mean.numpy(), z_log_var.numpy()], axis=1)
        dataset_labels = np.concatenate([y_train, y_test], axis=0)
        dataset_labels.transpose()

        data = np.column_stack((latent_features, dataset_labels))
        data_train = data[0:len_train, :]
        data_teste = data[len_train:, :]

        # save_file('mnist_train', data_train, 'csv')
        # save_file('mnist_test', data_teste, 'csv')

        return data_train, data_teste