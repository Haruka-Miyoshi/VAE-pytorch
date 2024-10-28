# **Variational Auto-Encoder**<b>
VAE（Variational Autoencoder，変分オートエンコーダ）は，生成モデルの一種で，特にデータの潜在的な構造を学習し，それをもとに新しいデータを生成することが得意な機械学習モデルです．VAEは，従来のオートエンコーダ（Autoencoder）を拡張したモデルで，データの生成に応用できるという特徴を持っています．本レポジトリは，VAEのPytorch実装を含みます．`main.ipynb`では，`minist dataset`を使用したVAEの評価を行っています．

---

A Variational Autoencoder (VAE) is a type of generative model particularly well-suited for learning the underlying structure of data and generating new data based on that structure. VAE is an extension of the traditional Autoencoder model, with the added ability to generate data. This makes it especially useful for applications in data generation. This repository includes a PyTorch implementation of VAE. In `main.ipynb`, the VAE is evaluated using the MNIST dataset.

## **Reference**
```
@misc{kingma2022autoencodingvariationalbayes,
      title={Auto-Encoding Variational Bayes}, 
      author={Diederik P Kingma and Max Welling},
      year={2022},
      eprint={1312.6114},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/1312.6114}, 
}
```
