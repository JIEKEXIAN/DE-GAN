# DE-GAN：Domain Embedded GAN for High Qualit Face Image Inpainting

# Introduction

Existing face inpainting methods incorporate only one type of simple facial features for face completion, and their results are still undesirable. To improve face inpainting quality, we propose a Domain Embedded Generative Adversarial Network (DE-GAN) for face inpainting.
DE-GAN embeds three types of face domain knowledge (i.e., face mask, face part, and landmark image) via a hierarchical variational auto-encoder (HVAE) into a latent variable space to guide face completion. Two adversarial discriminators, a global discriminator and a patch discriminator, are used to judge whether the generated distribution is close to the real distribution or not.
Experiments on two public face datasets demonstrate that our proposed method generates higher quality inpainting results with consistent and harmonious facial structures and appearance than existing methods and achieves the state-of-the-art performance, esp. for inpainting large-pose side faces.

