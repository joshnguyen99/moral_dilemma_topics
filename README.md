# Mapping Topics in 100,000 Real-life Moral Dilemmas

This repository contains the data and code for our ICWSM 2022 Paper, _Mapping Topics in 100,000 Real-life Moral Dilemmas_.

## Dataset

Instructions on downloading the dataset can be found in [`dataset/README.md`](dataset/README.md).

## Setting up a Python environment

We will be using Anaconda for setting up a virtual environment.

```sh
# Create a new environment
conda create --name icwsm2022 python=3.8.5

# Install packages and dependencies
pip install -r requirements.txt
```

## Topic modeling

We use two topic models, Latent Dirichlet Allocation and Nonnegative Matrix Factorization. To use then, first `cd` to `scripts`. The documentation can be found in that folder.

## Citation

Please use the following BibTex entry to cite our work.

```bibtex
@article{Nguyen_Lyall_Tran_Shin_Carroll_Klein_Xie_2022, 
    title={Mapping Topics in 100,000 Real-Life Moral Dilemmas},
    author={Nguyen, Tuan Dung and Lyall, Georgiana and Tran, Alasdair and Shin, Minjeong and Carroll, Nicholas George and Klein, Colin and Xie, Lexing},
    journal={Proceedings of the International AAAI Conference on Web and Social Media},
    volume={16},
    number={1},
    year={2022},
    pages={699-710}
}
```