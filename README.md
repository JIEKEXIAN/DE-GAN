# DE-GAN：Domain Embedded GAN for High Qualit Face Image Inpainting

# Introduction

Existing face inpainting methods incorporate only one type of simple facial features for face completion, and their results are still undesirable. To improve face inpainting quality, we propose a Domain Embedded Generative Adversarial Network (DE-GAN) for face inpainting.
DE-GAN embeds three types of face domain knowledge (i.e., face mask, face part, and landmark image) via a hierarchical variational auto-encoder (HVAE) into a latent variable space to guide face completion. Two adversarial discriminators, a global discriminator and a patch discriminator, are used to judge whether the generated distribution is close to the real distribution or not.
Experiments on two public face datasets demonstrate that our proposed method generates higher quality inpainting results with consistent and harmonious facial structures and appearance than existing methods and achieves the state-of-the-art performance, esp. for inpainting large-pose side faces.

The main contributions are summarized as follows:

* DE-GAN incorporates an HVAE in the generator to embed three types of face domain information into latent variables as the guidance for face inpainting, which produces more natural faces.

* Item Different from existing face inpainting methods that use multiple stages to incorporate prior information to complete image or face inpainting, which cannot be end-to-end trained, our proposed method is end-to-end trainable.

* Item To the best of our knowledge, our work is the first on the evaluation of the large-pose side-face inpainting problem. Our inpainting method achieves the state-of-the-art visual quality and facial structures for inpainting large-pose side faces.

# The Structure of DEGAN

![structure_DEGAN](G:\github\GitHub\DE-GAN\fig\structure_DEGAN.png)

# Getting Started

## Requirements:

This code was tested with Tensorflow 1.14 , Keras 2.2.4, CUDA 10.1, Python 3.7 Ubuntu 16.04

* Install tensorflow 1.14

* install keras 2.2.4

  ### Clone this repo:

  ```
  git clone git@github.com:JIEKEXIAN/DE-GAN.git
  cd DE-GAN
  ```

  ## Training
  
  ```
  python train.py
  ```
  
  notice: you need to set train dataset path, valid dataset path and save path.
  
  ## Pretrained Models

DE-GAN's model parameters have been put into Baidu web disk.You can download the test at this link.

link：https://pan.baidu.com/s/1dxqDb_CVW_OBRwQCfqIECA 
code：4ini

## Test

````
python test.py -p image path -s save path
````

