# Generative AI Experiment Principle

## 1. VAE (Variational Autoencoder)

### Core Idea
VAE learns a probabilistic mapping from input data to a latent space, enabling generation by sampling from the learned distribution.

### Mathematical Formulation
- Encoder: q(z|x) → learns approximate posterior
- Decoder: p(x|z) → reconstructs input from latent
- Loss = Reconstruction Loss + KL Divergence

### Key Components
- Reparameterization trick for gradient estimation
- Gaussian prior on latent space
- Stochastic sampling for generation

## 2. GAN (Generative Adversarial Network)

### Core Idea
Two networks compete: Generator tries to fool Discriminator, Discriminator tries to distinguish real vs fake.

### Training Dynamics
- Minimax game: min_G max_D V(D,G)
- Mode collapse: generator produces limited variety
- Training instability: balancing D and G

### DCGAN Architecture
- Convolutional layers with batch norm
- ReLU/LeakyReLU activations
- Transposed convolution for upsampling

## 3. Diffusion Models

### Core Idea
Gradually add noise to data, then learn to reverse the process.

### Forward Process
- q(x_t|x_{t-1}) = N(x_t; sqrt(1-β_t)x_{t-1}, β_t I)
- Iteratively add Gaussian noise

### Reverse Process
- p(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t^2 I)
- Neural network predicts mean and variance

### DDPM (Denoising Diffusion Probabilistic Model)
- Fixed variance schedule
- Predict noise directly
- Markov chain for generation

## 4. Evaluation Metrics

### FID (Fréchet Inception Distance)
- Measures similarity of two distributions
- Uses InceptionV3 activations
- Lower = better quality

### IS (Inception Score)
- Measures diversity and quality
- Higher = better

### Precision/Recall
- Precision: fraction of generated samples that are valid
- Recall: fraction of real samples that can be generated

## 5. Experimental Design

### VAE Experiment
- Dataset: MNIST
- Latent dimension: 32
- Training: 50 epochs, Adam optimizer

### GAN Experiment
- Dataset: CIFAR-10
- Latent dimension: 100
- Training: 100 epochs, Adam optimizer (β1=0.5)

### Diffusion Experiment
- Model: Pre-trained DDPM (CelebA-HQ)
- Inference: 50 steps
- DDIM for faster sampling