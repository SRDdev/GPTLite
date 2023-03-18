# GPTLite
GPTLite is a PyTorch implementation of the GPT (Generative Pre-trained Transformer) language model. It is trained on the Shakespear dataset, and is built from scratch.

## Introduction to GPT
GPT (Generative Pre-trained Transformer) is a language model developed by OpenAI. It is based on the transformer architecture, which was introduced in the paper "Attention is All You Need" by Google researchers. The key idea behind GPT is to pre-train a deep neural network on a large dataset, and then fine-tune it on a specific task, such as language translation or question answering.

GPT's architecture consists of an encoder and a decoder, both of which are made up of multiple layers of self-attention and feed-forward neural network. The encoder takes in the input sequence and produces a representation of it, while the decoder generates the output sequence based on the representation.

GPT-2, an updated version of GPT, was trained on a dataset of over 40 GB of text data, and is able to generate human-like text, complete tasks such as translation and summarization, and even create original content.

GPTLite is a smaller version which is built for fine-tuning and is trained on the [Dataset](), which is still powerful enough to generate human-like text, but with less computational resources required.


## Usage
To fine-tune GPTLite on your specific task, you will need to provide your own dataset and adjust the fine-tuning script accordingly. The code for the model and the training script are provided in this repository.

## Details
```
Dataset : -----
FrameWork : Pytorch
Training : Google Colab
Epochs : 5000
Params : 10.788929 Million
```

## Requirements
- Pytorch
- GPUs

## Inference 
> [SriptGPT-small](https://huggingface.co/SRDdev/ScriptGPT-small) 

> [Sript_GPT](https://huggingface.co/SRDdev/Script_GPT) 


## References
- "Attention is All You Need" by Ashish Vaswani, et al. (https://arxiv.org/abs/1706.03762)
- "Language Models are Unsupervised Multitask Learners" by Alec Radford, et al. (https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Video](https://youtu.be/kCc8FmEb1nY)
