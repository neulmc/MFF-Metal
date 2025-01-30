# Lightweight metal surface defect segmentation method based on multiscale feature fusion and knowledge distillation
This is the source code of "Lightweight metal surface defect segmentation method based on multiscale feature fusion and knowledge distillation". 

### Introduction:

We propose a deep learning method named MFF-Metal based on multi-scale feature fusion and knowledge distillation for defect segmentation in metal images. Regarding the structure, it is an encoder-decoder framework and consists of multi-branch dilation bottleneck, richer feature fusion module, and a multi-scale attention decoder. Accord-ing to the appearance characteristics of the surface defects, the method integrates pix-el-level cross-entropy loss, region-based Dice loss, and boundary loss with shape in-formation. Moreover, considering the deployment requirements of defect segmenta-tion, a knowledge distillation strategy is adopted to compress the large-scale model in-to student model, further reducing the parameter numbers. In multiple experiments, compared to traditional convolutional networks and transformer segmentation models, our method achieves a higher mIoU score of 0.8771 and exhibits a significant ad-vantage in model spatial complexity, fully demonstrating the effectiveness and poten-tial application prospects of the method.

### Prerequisites

- pytorch >= 1.12.1(Our code is based on the 1.12.1)
- numpy >= 1.21.6

### Train and Evaluation
1. Clone this repository to local.

2. Download the public dataset NEU-Seg from the official sources or the link we provided https://pan.baidu.com/s/1ON8Q8gum0WUxR9R8yoZwgg?pwd=7zt.

3. Execute file 'build_boundary.py' to construct boundary annotations from the provided pixel level labels.

4. Execute the training file train.py until the termination condition is reached (training epochs is 100).
   (Option) Turn off knowledge distillation by switching the value of 'T_KD' to false in the code
   (Option) For comparison, we also provide implementation code for existing methods such as UNet, UNet++, AttentionUNet, TransUNet, etc.

During and after training, the predictions and checkpoints are saved and the "log_file" is constructed for recording losses and performances.

### Dataset
This dataset is sourced from the Global Artificial Intelligence Algorithm Elite Open Challenge hosted by Chinese Jiangsu Artificial Intelligence Society [http://bdc.saikr.com/c/rl/50185]. 
The dataset partitioning strictly follows the requirements of the competition group. 
We have achieved the highest level of awards among over 400 teams from around the world. 
This code repository is an extension of our work in the open challenge competition.
In the paper, we conducted more comparative experiments and analytical discussions to further demonstrate the effectiveness and innovation of the method.

### Results on NEU-Seg Dataset
|Method | Inclusion |  Patch |  Scratch |  mIoU | 
|:-----|:------:|:-----:| :-----:| :-----:| 
|UNet |  0.7778 |  0.9080 |  0.8504 |  0.8454 | 
|UNet++  | 0.7854 |  0.9131 |  0.8575 |  0.8520 | 
|AttentionUNet |  0.7833 |  0.9108 |  0.8524 |  0.8488 | 
|TransUNet |  0.7649  | 0.9023 |  0.8412  | 0.8361 | 
|LMFFNet |  0.8103 |  0.9257 |  0.8625 |  0.8662 | 
|MFF-Metal  | 0.8295  | 0.9257  | 0.8762 |  0.8771 | 

### Note
Executing knowledge distillation requires a teacher model, and existing teacher models are trained with a channel count of 180. 
The link is shown below https://pan.baidu.com/s/18ZSIb81WAn_AFVmS3pR4OQ?pwd=mwjs.
In actual deployment and implementation scenarios, the corresponding model can be flexibly selected according to the specific situation.

### Final models
This is the final model and log file in our paper. We used this model to evaluate. You can download by:
https://pan.baidu.com/s/1crHmNQxl4ZegfMBWPFkvSw?pwd=jwna code: jwna.

### References
[1] <a href="https://github.com/Greak-1124/LMFFNet">LMFFNet: A Well-Balanced Lightweight Network for Fast and Accurate Semantic Segmentation.</a>

[2] <a href="https://github.com/Beckschen/TransUNet">TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation.</a>

[3] <a href="https://arxiv.org/abs/1505.04597">U-Net: Convolutional Networks for Biomedical Image Segmentation.</a>

