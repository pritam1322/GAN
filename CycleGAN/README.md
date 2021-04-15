**CycleGAN**

# Introduction

CycleGAN uses a unique approach for learning to translate an image from
a source domain X to a target domain Y in the absence of paired examples
was used in CycleGAN. It also uses inverse functions to get the original
image back.
CycleGAN uses the 6/9 residual block architecture for generator similar to Johnson et al. and PatchGAN of 70 X 70 for discriminator.

![image.png](https://github.com/pritam1322/GAN/blob/main/CycleGAN/image.png)


# Loss
Total loss = Adversarial loss + Cycle consistency = L(G, F, DX, DY ) =LGAN(G, DY , X, Y ) + LGAN(F, DX, Y, X) + λLcyc(G, F),  

# Dataset
 The dataset used in the project i staken from VOC Dataset.

# Reference- 
@Aladdin perrson 
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/CycleGAN
