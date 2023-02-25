# diffusion-models: sample implementations of diffusion models in PyTorch

This repository contains sample implementations of diffusion models in PyTorch. The implementations are not production ready, and primarily focus on simplicity and ease of understanding. Sampling from the model is currently slow: on my setup with a RTX 3060, generating 10k samples for calculating FID on the CIFAR10 test set takes about an hour. 

Currently only the [DDPM model](https://arxiv.org/pdf/2006.11239.pdf) is implemented.

## Install

All dependencies can be installed with conda using

```
conda env create -f conda/environment.yml
```

You can also perform an editable installation via

```
conda env create -f conda/environment.yml
conda activate diff-models
pip install -e .
```

The installation can be tested with either

```
make test_train_fashion_mnist
```

or

```
make test_train_cifar10
```

which should train a model on the specified dataset for 10 epochs.

## Train a Diffusion Model

I don't currently have a config file/make recipe to train a model to convergence for the Fashion MNIST or CIFAR10 datasets (although the model that results from `make test_train_fashion_mnist` produces pretty good samples already). My guess is that the hyperparameters in the config files should work well.

For now, if you want to adjust the training parameters, you can adjust the configuration in the `config` folder manually or run the training scripts with the desired command line arguments.

Note that training an unconditional generative model on CIFAR10 to convergence may take a long time; the DDPM paper reported it took approximately 10 hours to train a model to convergence on the equivalent of 8 V100 GPUs (TPU v3-8).

## Acknowledgements

Parts of the implementation are based off [Phil Wang's PyTorch DDPM implementation](https://github.com/lucidrains/denoising-diffusion-pytorch) and the [original Tensorflow implementation](https://github.com/hojonathanho/diffusion) by Jonathan Ho and Ajay Jain.

