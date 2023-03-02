# neuroGPT
NeuroGPT is an open-source project that provides an easy-to-use framework for training and generating text using the GPT architecture. With support for both CPU and GPU training, this project allows users to quickly and easily generate text in a variety of contexts, from literature to machine learning.

The project includes pre-trained models for Shakespearean literature and OpenWebText, as well as instructions for training your own models using PyTorch and HuggingFace's transformers library. The project also includes support for logging and tracking training progress using Weights and Biases.

Whether you're a researcher looking to experiment with new language models or a writer looking for a tool to help generate new ideas, NeuroGPT is a powerful and flexible framework that can help you achieve your goals.

## Installation
To run the code, you need to install the following dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- ```pip install transformers``` for huggingface transformers version <3 (required to load GPT-2 checkpoints)
- ```pip install datasets``` for huggingface datasets version <3 (needed to download and preprocess OpenWebText)
- ```pip install tiktoken``` for OpenAI's fast BPE code version <3
- ```pip install wandb``` for optional logging version <3
- ```pip install tqdm```

## Quick Start
If you want to try out the code without much hassle, you can train a character-level GPT model on Shakespeare's works. First, download the text file as a single (1MB) file and turn it into a stream of integers by running:

```
$ python data/shakespeare_char/prepare.py
```  

This generates train.bin and val.bin files in the data directory. Now, you can train the GPT model. The model size depends on the resources available on your system:

### Training on GPU
If you have a GPU, you can train a baby GPT model with the settings in the config/train_shakespeare_char.py file. Run the following command:

```
$ python train.py config/train_shakespeare_char.py
```  

This trains a GPT model with a context size of up to 256 characters, 384 feature channels, and a 6-layer Transformer with 6 heads in each layer. On an A100 GPU, the training takes around 3 minutes, and the best validation loss is 1.4697. The model checkpoints are saved in the --out_dir directory out-shakespeare-char. Once training is done, you can generate samples using the best model by running:

```
$ python sample.py --out_dir=out-shakespeare-char
```  

### Training on CPU
If you don't have a GPU, you can still train a GPT model, but with lower settings. We recommend using the latest PyTorch nightly version to make your code more efficient. Run the following command:

```
$ python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```  

Here, we are running the model on a CPU, so we need to set --device=cpu and disable PyTorch 2.0 compile with --compile=False. We evaluate the model after every 20 iterations (--eval_iters=20), use a context size of 64 characters (--block_size=64), and a batch size of 12 (--batch_size=12). The model is a smaller 4-layer Transformer with 4 heads in each layer and an embedding size of 128. We train for 2000 iterations and decay the learning rate after 2000 iterations with --lr_decay_iters. We also use less regularization with --dropout=0.0. This takes about 3 minutes to train, and the validation loss is 1.88. After training, you can generate samples using the best model by running:

```
$ python sample.py --out_dir=out-shakespeare-char
```  

#### Tips for Apple Silicon Macbooks
If you're using an Apple Silicon Macbook, you can use the `--device mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks.
