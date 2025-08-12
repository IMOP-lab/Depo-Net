# Harmonized spatial-frequency domain synergy driven geospatial feature synthesis for enhanced SAR semantic segmentation

### [Project page](https://github.com/IMOP-lab/Depo-Net) | [Our laboratory home page](https://github.com/IMOP-lab) 

<div align=left>
  <img src="image_Depo_new.png">
</div>
<p align=left>
  Figure 1: Detailed network structure of the Depo-Net.
</p>

The Depo-Net architecture employs two distinct yet complementary encoder paths: the RepFViT path (Fig.\ref{image_Depo}(a)), tailored to capture localized spatial features, and the MaxViT path (Fig.\ref{image_Depo}(b)), suited for global dependency modeling across extended regions. Within RepFViT, the Spatio-Frequency Synergistic Modulation (SFSM) decomposes input frequency components, enabling high-frequency noise isolation and low-frequency structure preservation, thereby suppressing the effects of high-interference environments. The MaxViT path is configured for cross-scale global feature extraction, facilitating spatial association capture in complex scenes. In the feature fusion phase, a Dual-Stream Confluence (DSC) mechanism integrates multi-scale convolutional operations with a channel attention strategy, merging feature representations from both encoder paths. Subsequently, the Harmonized Subspace Spectro-Temporal Transformer (HastFormer) captures spatial correlations intrinsic to diverse terrain distributions. In the decoding stage, to mitigate the cumulative amplification of noise in layer-by-layer upsampling, Pluri-frequency Mamba (purfMamba) is introduced. Proposed purfMamba applies synergistic processing across the frequency and spatial domains, focusing on preserving coherence between global and local information at high resolution while maintaining noise control throughout the upsampling process.

## Installation
Initial learning rates are uniformly set at 0.0001, with batch sizes standardized to 2 across all models.  Experimental evaluations are conducted within a consistent hardware and software environment, featuring a workstation equipped with dual NVIDIA RTX 4080 Super GPUs, an AMD Ryzen R9-5950X processor, and 128GB RAM.  The experimental framework is structured in Python 3.9, employing PyTorch 2.0.0 and CUDA 11.8 for computational acceleration.

## Experiment
### Datasets
1.SARBuD 1.0 \url{https://github.com/CAESAR-Radi/SARBuD}.

2.HRSID \url{https://github.com/chaozhong2010/HRSID}.

3.FRBS \url{https://drive.google.com/file/d/15WYzzFZvAHmqSIW0PXXRTp_YVd_868l8/view}.

### Baseline
We provide GitHub links pointing to the PyTorch implementation code for all networks compared in this experiment here, so you can easily reproduce all these projects.

[U-Net](https://github.com/milesial/Pytorch-UNet);[ENet](https://github.com/davidtvs/PyTorch-ENet);[SegNet](https://github.com/vinceecws/SegNet_PyTorch?tab=readme-ov-file);[ICNet](https://github.com/hszhao/ICNet);[LEDNet](https://github.com/sczhou/LEDNet);[MRUNet](https://github.com/cyan-utokyo/MRUnet.git);[DconnNet](https://github.com/Zyun-Y/DconnNet);[PSPNet](https://github.com/hszhao/PSPNet.git);[UNet+Att](https://github.com/EvilPsyCHo/Attention-PyTorch.git);[PAttUNet](https://github.com/faresbougourzi/PDAtt-Unet);[DAttUNet](https://github.com/faresbougourzi/PDAtt-Unet);[DANet](https://github.com/junfu1115/DANet);[R2U-Net](https://github.com/ncpaddle/R2UNet-paddle); [Attention R2UNet](https://github.com/LeeJunHyun/Image_Segmentation);[poly pvt](https://github.com/DengPingFan/Polyp-PVT.git);[BAT](https://github.com/sharkdp/bat.git);[MDViT](https://github.com/siyi-wind/MDViT.git);[TransUNet](https://github.com/Beckschen/TransUNet.git);[UNeXt](https://github.com/jeya-maria-jose/UNeXt-pytorch.git);[TransFuse](https://github.com/Rayicer/TransFuse.git);[MambaHSI](https://github.com/li-yapeng/MambaHSI.git).

### Results
Table 1: The segmentation results of Depo-Net are compared with previous models on three SAR datasets (SARBuD, HRSID, FRBS), presented in three tables. The highest score for each metric is highlighted in red, while the second-best score is highlighted in blue. Segmentation models are detailed by method categories, with the best result within each category also highlighted in blue.

<div align=left>
  <img src="Tables/SARBud.jpg">
</div>
<p align=left>
   Segmentation results are presented on SARBuD data.
</p>

<div align=left>
  <img src="Tables/HRSID.jpg">
</div>
<p align=left>
   Segmentation results are presented on HRSID data.
</p>

<div align=left>
  <img src="Tables/FRBS.jpg">
</div>
<p align=left>
   Segmentation results are presented on FRBS data.
</p>

#### Visual segmentation results
A visual comparison of Depo-Net and other segmentation models is conducted across three SAR datasets. Five representative images are selected from each dataset for display. In these images, the segmentation results for target objects are filled in green, while the ground truth is outlined in red. Panel (a) represents the original image, {(b)-(g)} present the best segmentation results from each method category across the datasets, as shown in Table1. Specifically, (b) represents the best-performing model based on CNN methods across the three datasets, (c) represents the best-performing model based on CNN with attention mechanisms, and so on for the remaining categories, and (h) represents the segmentation result of Depo-Net.
<div align=left>
  <img src="Images/image_baseline_new.png">
</div>
<p align=left>
    Figure 2:Visual segmentation results
</p>

### Ablation study

#### Effect of Removing Module
Table 2: The ablation results of key modules in Depo-Net on SARBuD, with the optimal values highlighted in red, and the suboptimal values highlighted in blue. The last row of the table represents our proposed Depo-Net. Here, "\checkmark" indicates the presence of the module, and "Hasformer (DL)" denotes that the Hasformer is placed at the deepest layer of the network.

<div align=left>
  <img src="Images/Tables2.jpg">
</div>
<p align=left>
</p>
<div align=left>
  <img src="Images/Tables3.jpg">
</div>
<p align=left>
</p>
#### Integration of SFSM Across Multiple Architectures

Table 3: The comparison results of the baseline model with and without SFSM on SARBuD are presented, with the optimal values highlighted in red and the second-best values highlighted in blue. Here, \(\mathcal{F}\) denotes the integration of SFSM into the original baseline model.

<div align=left>
  <img src="Images/Tables4.jpg">
</div>
<p align=left>
</p>

#### Validation Across Different Structures in HastFormer

Table 4: In Hasformer, the query (Q), key (K), and value (V) in the self-attention mechanism are replaced with a combination of the real and imaginary parts of the Fourier transform and the wavelet transform. The last row in the table shows our design in Hasformer. The optimal values are highlighted in red, while the suboptimal values are highlighted in blue.

<div align=left>
  <img src="Images/Tables5.jpg">
</div>
<p align=left>
</p>

#### Feature analysis

<div align=left>
  <img src="Images/image_ab_fe_an.png">
</div>
<p align=left>
</p>

Feature structure maps from different ablation experiments. (a) Original input SAR image. (b) Decoding with the traditional Mamba structure. (c) Decoding with the purfMamba module, replacing the conventional Mamba structure. (d) Feature maps extracted using the U-Net structure. (e) Feature maps extracted by U-Net with SFSM integration. Panels (c) and (e) demonstrate significant improvements in noise artifact reduction and clearer structural contours compared to their respective baseline experiments.


