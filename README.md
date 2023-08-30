# SCALE SELECTION NETWORK WITH ATTENTION MECHANISM FOR CROWD COUNTING

Crowd counting is an important cornerstone in the tasks related to crowd analysis. In recent years, an increasing number of deep learning methods have been proposed in the field of image processing and achieved amazing results. However, there are some challenges need to be solved in the current crowd counting task: large-scale variations in scale and interference from the background. Both of them will lead to poor prediction results. Therefore, we propose a scale selection module to deal the scale variation problem in images. And for background interference, we proposed an attention module to reduce the interference of background information. Moreover, we evaluated our model on four commonly used datasets and compared the performance with other state of-the-art methods to demonstrate the competitiveness of our approach.

<img src="https://i.imgur.com/LBIfNME.png" alt="https://i.imgur.com/LBIfNME.png" title="https://i.imgur.com/LBIfNME.png" width="1312" height="350">

# Environment
- Python 3.8
- pytorch 1.12.1

Please run the follow line to install enviroment
```python

pip install -r requirements.txt

```

# How to try

## Download dataset (Places365、CelebA、ImageNet)
[ShanghaiTech](https://www.kaggle.com/datasets/tthien/shanghaitech)  (no official)

[UCF_CC_50](https://www.crcv.ucf.edu/data/ucf-cc-50/)

[UCF_QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/)

[NWPU](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)

## Data preprocess

Edit the root and cls in generate_density.py
```python

root = 'The root of image'
target_root = 'The root of saving generate ground-truth'
cls = 'For which dataset' # Ex. SHHA, NWPU, UCF_QNRF, UCF_CC_50

```

Run generate_density.py in data_preprocess to generate ground-truth density map


Please put the image and ground-truth in the same folder
```python

Data_root/
         -train/
               -IMG_1.h5
               -IMG_1.jpg
               ⋮
         -test/
               -IMG_1.h5
               -IMG_1.jpg
               ⋮
 ⋮

```

Run the data_pair.py to generate data_list
```python

python data_preprocess/data_pair.py

```


## Backbone pretrained model
["Here"](https://drive.google.com/drive/u/4/folders/1QeLZc7_4TZVZ7awRQXNmgvGtPl6OGUAR)

## Training
```python

python train.py --data_root 'data_root' --epochs 4000

```

## Run testing
```python

python test.py --weight_path 'checkpoint_path'

```

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


