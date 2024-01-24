# Image Quality Assessment: Measuring Perceptual Degradation via Distribution Measures in Deep Feature Spaces
----------------------------
This is the repository of the paper [Image Quality Assessment: Measuring Perceptual Degradation via Distribution Measures in Deep Feature Spaces](xxx). 

Three deep distribution measures are proposed, **the DeepWSD**, **the DeepJSD** and **the DeepSKLD**. The default form is based on the VGG architecture. Other variants with the SqueezeNet, the MobileNet, and the ResNet are also proposed. 

## Advantages of deep network based distribution measures:
1.  Superior performance on synthetic distortion-based datasets without further fine-tuning.

2.  Differentiability in guiding perceptual image enhancement.

3.  Adaptivity to diverse network architectures.

-----------------------------
## Updating log:
2024/1/24: the repository is created, and the quality assessment result and the image-to-image enhancement results are uploaded in 'results' folder. 

-----------------------------
## Requirements:


------------------------------

## Useage:
Please compare reference and distorted images one by one. The batch-based computation **is not supported**.  

------------------------------

## Acknowledgement:
We thanks a lot for work of 'dingkeyan93' and work of [DISTS](https://github.com/dingkeyan93/DISTS). DeepWSD is mostly inspired by the insightful idea from him. 
