# GAN-nam-Style

This project is aimed at performing style transfer on music using models designed from scratch. Instead of the usual MIDI format, we want to work on the raw music spectrograms for more universality and flexibility.

We use [this great dataset](https://github.com/mdeff/fma/)

## Architecture

Our main architecture is an RNN-AE that uses multiple layers of LSTMs around the latent space. Another research models sandwiches this model between two dimension-reducing/increasing convolutive layers.

We optimize these models using an MSELoss. The results are great when visualy comparing the spectrograms but, while the regenerated music does have melodic and rythmic similarities to the input one, it has a lot of noise.

After trying some variations of our original model, we think that the main fault for this noise comes from the MSELoss, which does not make the model optimize for human-distinction. Having a small, but non-zero, intensity over a large band of frequencies translates into a huge white noise while only generating a small MSELoss

A possible future solution would be to couple this MSELoss with an adversarial model concurrently train to distinguish heavily-noised samples from real music.
