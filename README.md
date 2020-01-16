# Adversarial Calibrated Loss for Semi-supervised Semantic Segmentation

In this repo, we use the output of the segmentation network as the pseudo label based on the supervisory signal from the discriminator.
Pixels with higher confidence, which is the output of discriminator, mean that pixels are more like real label, so we give more weights to pixels with higher confidence. 
Because the confidence is not well calibrated to the real accuracy, temperature scaling is conducted.

In addition, Virtual Adversarial Training, which makes classification network be more robust to adversarial noise, is applied to the segmentation network. We use confidence filtering so that the pixels with high confidence are only used for VAT.
The code are heavily borrowed from Adversarial Learning for Semi-supervised Semantic Segmentation([Link](https://github.com/hfslyc/AdvSemiSeg)), Temperature Scaling ([Link](https://github.com/gpleiss/temperature_scaling)), and VAT ([Link](https://github.com/9310gaurav/virtual-adversarial-training))


## Prerequisite

* CUDA/CUDNN
* Torchvision
* Numpy
* pytorch == 0.4.0 
* python-opencv == 3.4.2 
