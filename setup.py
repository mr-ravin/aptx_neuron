import os
from setuptools import setup, find_packages

# Read long description safely
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read() if os.path.exists("README.md") else ""

setup(
    name="aptx_neuron",
    version="0.0.2",
    author="Ravin Kumar",
    author_email="mr.ravin_kumar@hotmail.com",
    description="A PyTorch implementation of the APTx Neuron.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mr-ravin/aptx_neuron",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research"
    ],
    keywords=[
        "APTx Neuron", "unified neuron", "neuron", "activation function", "deep learning", "pytorch", "neural network", "perceptron"
        "machine learning", "artificial intelligence", "AI", "ML", "DL", "torch"
    ],
    license="MIT",
    include_package_data=True
)
