# Rotation Equivariant Deforestation Segmentation

This is a PyTorch implementation of a rotation equivariant U-Net model for segmentation and classification of deforestation areas as described in our paper:

https://arxiv.org/abs/2110.13097

## Installation

```
conda create --name roteqseg python=3.7.9
conda activate roteqseg
python setup.py install
````

## Data

The dataset used in this project is from https://stanfordmlgroup.github.io/projects/forestnet/ and can be dowloaded from their page.

Download the dataset into the roteqseg folder and unzip:

```
unzip ForestNetDataset.zip
```

## Run the examples

We provide both a non-rotation equivariant model and a rotation equivariant model to allow a comparison to be made between the two models.

```
cd roteqseg
mkdir Outputs
python trainer.py --savedir 'code_test' --epochs 12 --model 'unet'
python trainer.py --savedir 'eq_code_test' -- epochs 12 --model 'unet_eq'
```

We also provide a notebook file, Post_Run_Analysis.ipynb, which performs the post processing we used to generate figures and numeric results used in the paper.

## Cite

Please cite our paper if you make use of our work:

```
@article{mitton2021rotation,
  title={Rotation Equivariant Deforestation Segmentation and Driver Classification},
  author={Mitton, Joshua and Murray-Smith, Roderick},
  journal={NeurIPS 2021 Workshop on Tackling Climate Change with Machine Learning},
  year={2021}
}
```


