# SCALE SELECTION NETWORK WITH ATTENTION MECHANISM FOR CROWD COUNTING

Crowd counting is an important cornerstone in the tasks related to crowd analysis. In recent years, an increasing number of deep learning methods have been proposed in the field of image processing and achieved amazing results. However, there are some challenges need to be solved in the current crowd counting task: large-scale variations in scale and interference from the background. Both of them will lead to poor prediction results. Therefore, we propose a scale selection module to deal the scale variation problem in images. And for background interference, we proposed an attention module to reduce the interference of background information. Moreover, we evaluated our model on four commonly used datasets and compared the performance with other state of-the-art methods to demonstrate the competitiveness of our approach.

<img src="https://i.imgur.com/LBIfNME.png" alt="https://i.imgur.com/LBIfNME.png" title="https://i.imgur.com/LBIfNME.png" width="1312" height="350">

# Environment
- Python 3.7.0
- pytorch
- opencv
- PIL  
- colorama

or see the requirements.txt

# How to try

## Download dataset (Places365、CelebA、ImageNet)
[Places365](http://Places365.csail.mit.edu/)  
[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
[FFHQ](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP)
[Paris Street View](https://github.com/pathak22/context-encoder/issues/24) (Non-publicise, please ask the paper author)

## Set dataset path

Edit txt/xxx.txt (set path in config)
```python

data_path = './txt/train_path.txt'
mask_path = './txt/train_mask_path.txt'
val_path = './txt/val_path.txt'
val_mask_path = './val_mask_file/' # path
test_path: './txt/test_path.txt'
test_mask_1_60_path: './test_mask_1+10_file/' # path

```

txt example
```python

E:/Places365/data_256/00000001.jpg
E:/Places365/data_256/00000002.jpg
E:/Places365/data_256/00000003.jpg
E:/Places365/data_256/00000004.jpg
E:/Places365/data_256/00000005.jpg
 ⋮

```

**You can refer to our example.txt in the txt path**

## Preprocessing  
In this implementation, masks are automatically generated by ourself. stroke masks mixed randomly to generate proportion from 1% to 60%.

strokes (from left to right 20%-30% 30%-40% 40%-50% 50%-60%)
<img src="https://imgur.com/m3CStkN.png" alt="https://imgur.com/m3CStkN.png" title="https://imgur.com/m3CStkN.png" width="1000" height="200">

## Pretrained model
["Here"](https://drive.google.com/drive/u/4/folders/1QeLZc7_4TZVZ7awRQXNmgvGtPl6OGUAR)

## Run training
```python

python train.py (main setting data_path/mask_path/val_path/val_mask_path/batch_size/train_epoch)

```
1. set the config path ('./config/model_config.yml')
2. Set path and parameter details in model_config.yml

Note: If the training is interrupted and you need to resume training, you can set resume_ckpt and resume_D_ckpt.

## Run testing
```python

python test.py (main setting test_ckpt/test_path/test_mask_1_60_path/save_img_path)

```
1. set the config path ('./config/model_config.yml')
2. Set path and parameter details in model_config.yml

## Quantitative comparison

- Places365 

<img src="https://i.imgur.com/rU4n2cG.png" width="1312" height="350">

Quantitative evaluation of inpainting on Places365 dataset. We report Peak signal-to-noise ratio (PSNR), structural similarity (SSIM), Learned Perceptual Image Patch Similarity (LPIPS) and Frechet Inception ´ Distance (FID) metrics. The ▲ denotes larger, and ▼ denotes lesser of the parameters compared to our proposed model. (Bold means the 1st best; Underline means the 2nd best; † means higher is better; ¶ means lower is better)

- CelebA 

<img src="https://i.imgur.com/hfnk1QZ.png" width="1312" height="350">

Quantitative evaluation of inpainting on CelebA dataset. We report Peak signal-to-noise ratio (PSNR), structural similarity (SSIM), Learned Perceptual Image Patch Similarity (LPIPS) and Frechet Inception Distance ´ (FID) metrics. The ▲ denotes larger, and ▼ denotes lesser of the parameters compared to our proposed model. (Bold means
the 1st best; Underline means the 2nd best; † means higher is better; ¶ means lower is better)

- FFHQ 

<img src="https://i.imgur.com/C1DTqt2.png" width="1312" height="200">

 Quantitative evaluation of inpainting on FFHQ dataset. We report Peak signal-to-noise ratio (PSNR), structural similarity (SSIM), Learned Perceptual Image Patch Similarity (LPIPS) and Frechet Inception Distance ´ (FID) metrics. (Bold means the 1st best; † means higher is better; ¶ means lower is better; S means 5% to 20% mask
range; M means 21% to 40% mask range; L means 41% to 60% mask range)

- Paris Street View 

<img src="https://i.imgur.com/MHf8WQX.png" width="1312" height="200">

Quantitative evaluation of inpainting on Paris Street View dataset. We report Peak signal-to-noise ratio (PSNR), structural similarity (SSIM), Learned Perceptual Image Patch Similarity (LPIPS) and Frechet Inception ´ Distance (FID) metrics. (Bold means the 1st best; † means higher is better; ¶ means lower is better; S means 5% to
20% mask range; M means 21% to 40% mask range; L means 41% to 60% mask range)

- Cross dataset evaluation (Training on Places365 / Testing on CelebA)

<img src="https://i.imgur.com/mKMiMyX.png" width="1312" height="300">

Cross dataset evaluation of inpainting (training on Places365 dataset and testing on CelebA dataset. We report Peak signal-to-noise ratio (PSNR), structural similarity (SSIM). (Bold means the 1st best; Underline means the 2nd best; † means higher is better)

- Cross dataset evaluation (Training on CelebA / Testing on Places365)

<img src="https://i.imgur.com/fPvofNz.png" width="1312" height="250">

Cross dataset evaluation of inpainting (training on CelebA dataset and testing on Places365 dataset. We report Peak signal-to-noise ratio (PSNR), structural similarity (SSIM). (Bold means the 1st best; Underline means the 2nd best; † means higher is better)

All training and testing base on same 3090.

## Qualitative comparisons

- Places365

<img src="https://i.imgur.com/1MAfYLF.jpg" width="1000" style="zoom:100%;">

The generated image comparison of our method and all SOTA methods on Places365 dataset. From left to right are ground truth image, input image, CA, RW, DeepFill-V2, HiFill, Iconv, CRFill, AOT-GAN, TFill, SWMHT-Net, FcF, ESWT-Net.

- CelebA

<img src="https://i.imgur.com/nc19VK8.png" width="1000" style="zoom:100%;">

The generated image comparison of our method and all SOTA methods on CelebA dataset. From left to right are ground truth image, input image, CA, RW, DeepFill-V2, Iconv, RF, CRFill, AOT-GAN, TFill, SWMHT-Net, FcF, ESWT-Net.

- FFHQ

<div align=center>
<img src="https://i.imgur.com/1TYkF3D.png" width="650" height="250">
</div>

The generated image comparison of our method and all SOTA methods on FFHQ dataset. From left to right are ground truth image, input image, CA, TFill, SWMHT-Net, ESWT-Net.

- Paris Street View

<div align=center>
<img src="https://i.imgur.com/2soManj.png" width="650" height="250">
</div>

The generated image comparison of our method and all SOTA methods on Paris Street View dataset. From left to right are ground truth image, input image, SN, RW, RFR, SWMHT-Net, ESWT-Net.

- Cross dataset evaluation (Training on Places365 / Testing on CelebA)

<img src="https://i.imgur.com/oq5eGxR.png" width="1000" style="zoom:100%;">

Cross-dataset image generation comparison of our method with all SOTA methods on the CelebA dataset. From left to right are ground truth image, input image, DeepFill-V2, RW, Iconv, HiFill, MADF, AOT-GAN, Lama, ESWT-Net.

- Cross dataset evaluation (Training on CelebA / Testing on Places365)

<img src="https://i.imgur.com/x3s0v2j.png" width="1000" style="zoom:100%;">

Cross-dataset image generation comparison of our method with all SOTA methods on the CelebA dataset. From left to right are ground truth image, input image, DeepFill-V2, RW, Iconv, MADF, AOT-GAN, Lama, ESWTNet.

## Ablation study

- Ablation study table

<div align=center>
<img src="https://i.imgur.com/IjlLw3j.png" width="650" height="250">
</div>

Ablation study of all modual we used with size 256×256 images on Places365 dataset. We report Peak signal-to-noise ratio (PSNR), structural similarity (SSIM). (Bold means the 1st best; Underline means the 2nd best; † means higher is better; V means included module; V∗ means included module and get results from this stage.)

- Ablation study Qualitative comparisons

<div align=center>
<img src="https://i.imgur.com/UFeJK0D.png" width="550" height="300">
</div>

The results of each ablation experiment are shown. There are respective removed modules at the bottom of each image. Among them, ESWT (coarse) represents the original model design but the coarse result of the first stage and ESWT (refine) represents the output refinement result of the second stage of the original model.

## Object removal

<div align=center>
<img src="https://i.imgur.com/tKALlyh.png" width="650" height="250">
</div>

Object removal (size 256×256) results. From left to right: Original image, mask, object removal result.


## Acknowledgement
This repository utilizes the codes of following impressive repositories   
- [ZITS](https://github.com/DQiaole/ZITS_inpainting)
- [LaMa](https://github.com/saic-mdal/lama)
- [CSWin Transformer](https://github.com/microsoft/CSWin-Transformer)
- [Vision Transformer](https://github.com/google-research/vision_transformer)
- [SWMHT-Net](https://github.com/bobo0303/LIGHTWEIGHT-IMAGE-INPAINTING-BY-STRIPE-WINDOW-TRANSFORMER-WITH-JOINT-ATTENTION-TO-CNN)
- [MSCSWT-Net](https://github.com/bobo0303/MSCSWT-Net)

---
## Contact
If you have any question, feel free to contact wiwi61666166@gmail.com


