# Makefile

# ----Fashion MNIST Rules----

test_train_fashion_mnist:
	python diffusion_models/scripts/train_fashion_mnist.py --config configs/test_train_fashion_mnist.json

clean_fashion_mnist_ckpts: $(wildcard checkpoints/fashion_mnist/*)
	rm -rf $^

clean_fashion_mnist_samples: $(wildcard samples/fashion_mnist/*)
	rm -rf $^

clean_fashion_mnist_runs: $(wildcard runs/fashion_mnist/*)
	rm -rf $^

clean_fashion_mnist: clean_fashion_mnist_ckpts clean_fashion_mnist_samples clean_fashion_mnist_runs

# ----CIFAR 10 Rules----

test_train_cifar10:
	python diffusion_models/scripts/test_train_cifar10.py --config configs/train_cifar10.json

test_eval_cifar10:
	diffusion_models/scripts/test_eval_cifar10.py --config configs/eval_cifar10.json

clean_cifar10_ckpts: $(wildcard checkpoints/cifar10/*)
	rm -rf $^

clean_cifar10_samples: $(wildcard samples/cifar10/*)
	rm -rf $^

clean_cifar10_runs: $(wildcard runs/cifar10/*)
	rm -rf $^

clean_cifar10: clean_cifar10_ckpts clean_cifar10_samples clean_cifar10_runs
