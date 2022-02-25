from unittest import TestCase
import gait_mapper
import gait_mapper.vae


class TestGaitMapper():
    """Suite of tests for gait_mapper.vae module."""

    def test_encoder_decoder(self):
        """Test if the vae module can generate correct architecture."""
        # define parameters for the model
        windowLength = 200
        degreeOfFreedom = 6
        latentFeatures = 6
        # create encoder and decoder network
        vae = gait_mapper.vae.VAE(
            windowLength, degreeOfFreedom, latentFeatures)
        # check if the number of parameters are correct
        expect_num_params_encoder = 32620
        expect_num_params_decoder = 18742
        assert vae.encoder.count_params() == expect_num_params_encoder
        assert vae.decoder.count_params() == expect_num_params_decoder
