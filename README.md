# Image Quality Assessment: Measuring Perceptual Degradation via Distribution Measures in Deep Feature Spaces
----------------------------
This is the repository of the paper [Image Quality Assessment: Measuring Perceptual Degradation via Distribution Measures in Deep Feature Spaces](https://ieeexplore.ieee.org/document/10577432). 

Three deep distribution measures are proposed: **the DeepWSD**, **the DeepJSD**, and **the DeepSKLD**. The default form is based on the VGG architecture. Other variants of SqueezeNet, MobileNet, and ResNet are also proposed. 

## Advantages of deep network-based distribution measures:
1.  Superior performance on synthetic distortion-based datasets without further fine-tuning.

2.  Differentiability in guiding perceptual image enhancement.

3.  Adaptivity to diverse network architectures.

-----------------------------
## Updating log:
2024/1/24: the repository is created, and the quality assessment result and the image-to-image enhancement results are uploaded in the 'results' folder. 

-----------------------------
## Requirements:
imageio==2.31.1

matplotlib==3.7.2

numpy==1.25.2

Pillow==10.0.0

POT==0.9.0

------------------------------

## Usage:
### For Quality Assessment:
Please compare reference and distorted images one by one.

    if __name__ == '__main__':
        from PIL import Image
        import argparse
        from utils import prepare_image
    
        parser = argparse.ArgumentParser()
        parser.add_argument('--ref', type=str, default='images/I47.png')
        parser.add_argument('--dist', type=str, default='images/I47_03_05.png')
        args = parser.parse_args()
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        ref = prepare_image(Image.open(args.ref).convert("RGB")).to(device)
        dist = prepare_image(Image.open(args.dist).convert("RGB")).to(device)
    
        model = DeepWSD().to(device)
        score = model(ref, dist, as_loss=False)
        print('score: %.4f' % score.item())

### For Perceptual image enhancement: Please Refer to 'recover.py'
------------------------------

## Acknowledgement:
We thank 'dingkeyan93' a lot for the work [DISTS](https://github.com/dingkeyan93/DISTS). His insightful ideas inspire deep distribution measures. 

------------------------------
## Citation
        @ARTICLE{10577432,
          author={Liao, Xingran and Wei, Xuekai and Zhou, Mingliang and Li, Zhengguo, and Kwong, Sam},
          journal={IEEE Transactions on Image Processing}, 
          title={Image Quality Assessment: Measuring Perceptual Degradation via Distribution Measures in Deep Feature Spaces}, 
          year={2024},
          volume={33},
          number={},
          pages={4044-4059},
          doi={10.1109/TIP.2024.3409176}}
