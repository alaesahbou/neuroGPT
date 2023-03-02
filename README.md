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

## Reproducing GPT-2 Results

To reproduce GPT-2 results, the first step is to tokenize the dataset. We will use the OpenWebText dataset, which is an open reproduction of OpenAI's (private) WebText. To download and tokenize the dataset, we can run the following command:

```
python data/openwebtext/prepare.py
```  

The above command creates two files, train.bin and val.bin, which hold the GPT-2 byte pair encoding (BPE) token IDs in one sequence, stored as raw uint16 bytes.
Next, we need to train the GPT-2 model. To reproduce GPT-2 (124M) results, we need at least an 8X A100 40GB node and run the following command:

```
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```  

This command runs for about four days using PyTorch Distributed Data Parallel (DDP) and brings down the loss to around 2.85.

It is important to note that a GPT-2 model just evaluated on OpenWebText gets a validation loss of around 3.11, but if we fine-tune it, it will come down to around 2.85 (due to an apparent domain gap), making the two models match.

If we have a cluster environment with multiple GPU nodes, we can run the following commands across two nodes:

Run on the first (master) node with example IP 123.456.123.456:

```
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
```  

### Run on the worker node:

```
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```  

If we do not have Infiniband, we should prepend NCCL_IB_DISABLE=1 to the above commands. It is also a good idea to benchmark the interconnect (e.g., iperf3). The checkpoints are periodically written to the --out_dir. We can sample from the model by running the following command:

```
python sample.py
```  

To train on a single GPU, we can run the following command:

We can find all the arguments in the train.py script. We will likely need to tune many of those variables depending on our needs.

### Baselines
OpenAI GPT-2 checkpoints allow us to set up some baselines for OpenWebText. We can obtain the following numbers by running the following commands:

```
python train.py eval_gpt2
python train.py eval_gpt2_medium
python train.py eval_gpt2_large
python train.py eval_gpt2_xl
```  

The following table shows the losses on the training and validation datasets:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

It's worth noting that GPT-2 was trained on a closed dataset called WebText, which was never released. OpenWebText is an open reproduction of this dataset, but it's not an exact match. This creates a domain gap between the two datasets. If you take the GPT-2 (124M) checkpoint and fine-tune it directly on OpenWebText, you can reach a loss of approximately 2.85. This becomes the more appropriate baseline for reproducing the dataset.

Fine-tuning is similar to training, except that we initialize from a pre-trained model and use a smaller learning rate. To fine-tune a GPT model on new text, you can use the tiny Shakespeare dataset as an example. First, run the prepare.py script in the data/shakespeare directory to download and render the dataset into train.bin and val.bin files, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText, this process will only take seconds. Fine-tuning can be done in just a few minutes on a single GPU. To run an example fine-tuning, execute the following command:

```
$ python train.py config/finetune_shakespeare.py
```  

This command will load the config parameter overrides in config/finetune_shakespeare.py, which initializes the model from a GPT2 checkpoint and trains it with a smaller learning rate. If you encounter out-of-memory errors, try decreasing the model size or the context length. The best checkpoint (with the lowest validation loss) will be saved in the out_dir directory, which is out-shakespeare by default, according to the config file. To generate text using the fine-tuned model, run the following command:

```
$ python sample.py --out_dir=out-shakespeare
```  

This will prompt the model to generate text, and the output will be similar to the following:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.
JULIET:
And if thou steal, thou shalt not sell thyself.
THEODORE:
I do not steal; I sell the stolen goods.
THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
``` 

Please note that the hyperparameters in the finetune_shakespeare.py config file have not been heavily tuned, so feel free to experiment with different values. To sample from a pre-trained GPT-2 model or a model you trained yourself, use the sample.py script. For example, the following command will sample from the largest available gpt2-xl model:

```
$ python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```  

If you want to prompt the model with text from a file, you can use the --start=FILE:prompt.txt option.

## License

This project is licensed under the MIT License.

## Team

- Alae-eddine Sahbou [alaesahbou](https://github.com/alaesahbou) 💼
- Othmane Gattouchi 
- Hamza Slahmi [hamzaslh](https://github.com/hamzaslh) 💼
