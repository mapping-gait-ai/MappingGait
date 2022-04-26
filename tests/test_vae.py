import unittest
from pathlib import Path
import numpy as np
from gait_mapper.vae import VAE

from . import data_folder

class TestGaitMapper(unittest.TestCase):
    """Suite of tests for gait_mapper.vae module."""

    def setUp(self):
        # define parameters for the model
        self.window_length = 200
        # degree_of_freedom
        self.dof = 6
        self.latent_features = 2
        self.data = Path(data_folder,
            "stored_groupsplit_withoutS01722_latentfeatures_2_frequency_20.npy")

    def test_encoder_decoder(self):
        """Test if the vae module can generate correct architecture."""
        # create encoder and decoder network
        vae = VAE(
            self.window_length, self.dof, self.latent_features)
        # check if the number of parameters are correct
        expect_num_params_encoder = 32620
        expect_num_params_decoder = 18742
        assert vae.encoder.count_params() == expect_num_params_encoder
        assert vae.decoder.count_params() == expect_num_params_decoder


    def test_training(self):
        # Prepare data
        raw_data = np.load(self.data)
        data = raw_data[:4800].reshape(-1, self.window_length, self.dof)

        # Train model
        vae = VAE(
            self.window_length, self.dof, self.latent_features)
        vae.compile(optimizer="adam")
        history = vae.fit(data, epochs=3, batch_size=2)
        self.assertIn('loss', history.history)
        self.assertIn('reconstruction_loss', history.history)
        self.assertIn('kl_loss', history.history)