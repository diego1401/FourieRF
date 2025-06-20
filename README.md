# FourieRF
## [Project page](https://www.lix.polytechnique.fr/~gomez/FourieRF/index.html) |  [Paper](https://arxiv.org/abs/2502.01405)
This repository contains a pytorch implementation for the paper: [FourieRF: Few-Shot NeRFs via Progressive Fourier Frequency Control](https://arxiv.org/abs/2502.01405). We present a simple yet efficient approach to tackle the few-shot NeRF problem. This repository is built on top of the code base introduced by the paper [TensoRF: Tensorial Radiance Fields](https://arxiv.org/abs/2203.09517).




https://github.com/user-attachments/assets/ca02d2e8-a6cb-4e1a-84d0-8c59cd6f76eb




## Installation

```
conda create -n FourieRF python=3.10 -y
conda activate FourieRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard scikit-learn plyfile matplotlib

```

## Dataset
* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [Forward-facing](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

## Quick Start
Correctly set the data path on the ```configs/fourier_blender.txt``` file with the ``` datadir```  argument. 
The training script is in `train.py`, to train a FourieRF on the orchids scene of the forward-facing dataset:

```
python train.py --config configs/fourier_blender.txt --number_of_views 4
```


## Citation
If you find our code or paper helps, please consider citing:
```
@misc{gomez2025fourierffewshotnerfsprogressive,
      title={FourieRF: Few-Shot NeRFs via Progressive Fourier Frequency Control}, 
      author={Diego Gomez and Bingchen Gong and Maks Ovsjanikov},
      year={2025},
      eprint={2502.01405},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.01405}, 
}
```
