import tensorflow as tf


class VAE(tf.keras.Model):
    """Variational autoencoder of the gait mapper."""

    def __init__(self, window_length=200, degrees_of_freedom=6, latent_features=6, alpha=0.1):
        """
        Initialize the variational autoencoder.

        Args:
            window_length (int): The unit length of input signal.
            degrees_of_freedom (int): Degrees of freedom included in the input signal.
            latent_features (int): Number of latent features to be encoded.
            alpha (int): Negative slope coefficient of the leaky version of a Rectified Linear Unit.

        Returns:
            Encoder and decoder networks.
        """
        super().__init__()
        self.window_length = window_length
        self.degrees_of_freedom = degrees_of_freedom
        self.latent_features = latent_features
        self.alpha = alpha
        self.encoder = self._encoder()
        # print summary of encoder model
        # self.encoder.summary()
        self.decoder = self._decoder()
        # print summary of encoder model
        # self.decoder.summary()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def train_step(self, data):
        """Customize the logic for one training step.

        This method overrides `tf.keras.Model.train_step`.
        For concrete examples of how to override this method see
        [Customizing what happends in fit](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).

        Args:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
            values of the `Model`'s metrics are returned. Example:
            `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        with tf.GradientTape() as tape:
            z, z_mean, z_var = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(
                z_var - tf.square(z_mean) - tf.exp(z_var) + 1
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @property
    def metrics(self):
        """Customize metrics.

        List `Metric` objects here to make `tf.keras.metrics.Metric.reset_states()`
        called automatically at the start of each epoch or `evaluate()`.
        If you don't implement this property, you have to call `reset_states()`
        yourself at the time of your choosing.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        """Calls the model on new inputs and returns the outputs as tensors.

        This method overrides `tf.keras.Model.call`, which is required when
        subclassing `tf.keras.Model`.

        Note: This method should not be called directly.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.

        Returns:
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        """
        z, z_mean, z_var = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(
            z_var - tf.square(z_mean) - tf.exp(z_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

    def _encoder(self):
        """Encoder of the gait mapper.

        Note: The output of model is latent features and their distribution.
        """
        input_layer = tf.keras.layers.Input(shape=(self.window_length, self.degrees_of_freedom))
        encoder = tf.keras.layers.Conv1D(
            64, 5, activation='relu', name='conv1')(input_layer)
        encoder = tf.keras.layers.MaxPooling1D(2)(encoder)
        encoder = tf.keras.layers.Conv1D(
            64, 3, activation='relu', name='conv2')(encoder)
        encoder = tf.keras.layers.MaxPooling1D(2)(encoder)
        encoder = tf.keras.layers.Conv1D(
            32, 3, activation='relu', name='conv3')(encoder)
        encoder = tf.keras.layers.MaxPooling1D(2)(encoder)
        encoder = tf.keras.layers.Flatten()(encoder)
        encoder = tf.keras.layers.Dense(16, name='dense1')(encoder)
        encoder = tf.keras.layers.Dense(8, name='dense2')(encoder)
        encoder = tf.keras.layers.Dense(8, name='dense3')(encoder)
        encoder = tf.keras.layers.LeakyReLU(self.alpha)(encoder)
        # distribution of latent space variable
        distribution_mean = tf.keras.layers.Dense(
            self.latent_features, name='mean')(encoder)
        distribution_variance = tf.keras.layers.Dense(
            self.latent_features, name='log_variance')(encoder)
        latent_encoding = tf.keras.layers.Lambda(self._sample_latent_features)(
            [distribution_mean, distribution_variance])
        encoder_model = tf.keras.Model(input_layer, [latent_encoding, distribution_mean, distribution_variance])

        return encoder_model

    def _decoder(self):
        """Decoder of the gait mapper.
        """
        input_layer = tf.keras.layers.Input(shape=(self.latent_features))
        decoder = tf.keras.layers.Dense(64)(input_layer)
        decoder = tf.keras.layers.Reshape((1, 64))(decoder)
        decoder = tf.keras.layers.Conv1DTranspose(
            16, 3, activation='relu')(decoder)
        decoder = tf.keras.layers.Conv1DTranspose(
            32, 5, activation='relu')(decoder)
        decoder = tf.keras.layers.UpSampling1D(5)(decoder)
        decoder = tf.keras.layers.Conv1DTranspose(
            64, 5, activation='relu')(decoder)
        decoder = tf.keras.layers.UpSampling1D(5)(decoder)
        decoder_output = tf.keras.layers.Conv1DTranspose(6, 6)(decoder)
        decoder_output = tf.keras.layers.LeakyReLU(self.alpha)(decoder_output)
        decoder_model = tf.keras.Model(input_layer, decoder_output)

        return decoder_model

    def _sample_latent_features(self, distribution):
        """
        Sample latent features in the latent space.
        The latent space is follows a Gaussian distribution, which is defiend by
        its mean and variance.

        Args:
            distribution (list): A list containing mean and variance of a Gaussian distribution.

        Returns:
            Tensors.
        """
        distribution_mean, distribution_variance = distribution
        batch_size = tf.shape(distribution_variance)[0]
        random = tf.keras.backend.random_normal(
            shape=(batch_size, tf.shape(distribution_variance)[1]))
        return distribution_mean + tf.exp(0.5 * distribution_variance) * random