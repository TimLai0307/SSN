# SCALE SELECTION NETWORK WITH ATTENTION MECHANISM FOR CROWD COUNTING

Crowd counting is an important cornerstone in the tasks related to crowd analysis. In recent years, an increasing number of deep learning methods have been proposed in the field of image processing and achieved amazing results. However, there are some challenges need to be solved in the current crowd counting task: large-scale variations in scale and interference from the background. Both of them will lead to poor prediction results. Therefore, we propose a scale selection module to deal the scale variation problem in images. And for background interference, we proposed an attention module to reduce the interference of background information. Moreover, we evaluated our model on four commonly used datasets and compared the performance with other state of-the-art methods to demonstrate the competitiveness of our approach.

<img src="https://github.com/TimLai0307/SSN/blob/main/vis/architecture.png" alt="https://github.com/TimLai0307/SSN/blob/main/vis/architecture.png" title="https://github.com/TimLai0307/SSN/blob/main/vis/architecture.png" width="2789" height="1310">

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


<img src="https://github.com/TimLai0307/SSN/blob/main/vis/comparison.png" width="1337" height="449">

Quantitative evaluation of inpainting on Places365 dataset. We report Peak signal-to-noise ratio (PSNR), structural similarity (SSIM), Learned Perceptual Image Patch Similarity (LPIPS) and Frechet Inception ´ Distance (FID) metrics. The ▲ denotes larger, and ▼ denotes lesser of the parameters compared to our proposed model. (Bold means the 1st best; Underline means the 2nd best; † means higher is better; ¶ means lower is better)


## Qualitative comparisons

- Visualize

<img src="https://github.com/TimLai0307/SSN/blob/main/vis/visual.png" width="1279" height="503">

The generated image comparison of our method and all SOTA methods on Places365 dataset. From left to right are ground truth image, input image, CA, RW, DeepFill-V2, HiFill, Iconv, CRFill, AOT-GAN, TFill, SWMHT-Net, FcF, ESWT-Net.


## Ablation study

- Ablation study 

<div align=center>
<img src="https://github.com/TimLai0307/SSN/blob/main/vis/ablation.png" width="546" height="195">
</div>

Ablation study of all modual we used with size 256×256 images on Places365 dataset. We report Peak signal-to-noise ratio (PSNR), structural similarity (SSIM). (Bold means the 1st best; Underline means the 2nd best; † means higher is better; V means included module; V∗ means included module and get results from this stage.)


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


