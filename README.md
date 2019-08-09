# Tangent-Normal Adversarial Regularization for Semi-Supervised Learning
A TensorFlow implementation for the paper:

[Tangent-Normal Adversarial Regularization for Semi-Supervised Learning](https://arxiv.org/abs/1808.06088)

[Bing Yu<sup>\*</sup>](https://scholar.google.com/citations?user=elxl0m8AAAAJ&hl=en), [Jingfeng Wu<sup>\*</sup>](https://uuujf.github.io/), [Jinwen Ma](http://www.is.pku.edu.cn/~jwma/), [Zhanxing Zhu](https://sites.google.com/view/zhanxingzhu/)

### Requirements ###
1. python 3.6
2. tensorflow 1.9.0
3. numpy

### Usage ###

#### Generate data ####
`python cifar10_to_numpy.py`

#### Obtain pre-trained VAE ####
The VAE checkpoint can be obtained via two ways:
- train by yourself: `python train_vae.py`
- download a pre-trained one: [without augmentation](https://drive.google.com/drive/folders/15lZuhizCh0qOJQM_NXcfXOMsCtHPRjPw), [with augmentation](https://drive.google.com/drive/folders/1RejmyS8oiS2M5EEEYn8otTTo1rM9aA22)

#### Play with TNAR ####
All of the parameters should be easy to understand by their naming conventions.
- Training: `python train_tnar.py --resume vae-checkpoint`
- Test: `python test_tnar.py --resume tnar-checkpoint`

### Hyperparameters and Performance ###
See the [paper](https://arxiv.org/abs/1808.06088).

### Notes ###
- [Bibtex](https://uuujf.github.io/papers/tnar/TNAR_CVPR_2019.bib), [Slides](https://uuujf.github.io/papers/tnar/slides.pdf), [Poster](https://uuujf.github.io/papers/tnar/poster.pdf)
- Some codes are borrowed from [vat_tf](https://github.com/takerum/vat_tf).
