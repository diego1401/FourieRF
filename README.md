# FourieRF
## [Project page](https://www.lix.polytechnique.fr/~gomez/FourieRF/index.html) |  [Paper (Coming Soon)]
This repository contains a pytorch implementation for the paper: [FourieRF: Few-Shot NeRFs via Progressive Fourier Frequency Control]. We present a simple yet efficient approach to tackle the few-shot NeRF problem. This repository is built on top of the code base introduced by the paper [TensoRF: Tensorial Radiance Fields](https://arxiv.org/abs/2203.09517).



https://github.com/user-attachments/assets/083fa6c2-958f-482c-bd44-81ee0830ff3b



## Installation

```
conda create -n FourieRF python=3.10 -y
conda activate FourieRF
pip install -r requirements.txt
```

## Dataset
* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [Forward-facing](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

## Quick Start
Correctly set the data path on the ```configs/fourier_example.txt``` file with the ``` datadir```  argument. 
The training script is in `train.py`, to train a FourieRF on the orchids scene of the forward-facing dataset:

```
python train.py --config configs/fourier_example.txt
```


## Citation
If you find our code or paper helps, please consider citing:
```
TODO
```
