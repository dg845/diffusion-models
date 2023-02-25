from setuptools import setup, find_packages

setup(
    name='diffusion-models',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'scipy',
        'tqdm',
        'einops',
    ],
)