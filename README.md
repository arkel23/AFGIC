# Awesome Fine-Grained Image Classification[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

I tried to condense the (main) contributions (or the used methodology) from each paper into a line or two to observe trends across years.

Also made a companion website on [GitHub Pages](https://arkel23.github.io/AFGIC/)
with summaries of all papers for a year + 1-slide summary of close to 200 surveyed papers.

Paper scraping description in [link](docs/literature_scraping.md).

If you have any problems, suggestions or improvements, please submit the issue or PR.

## Surveys

- Fine-Grained Image Analysis With Deep Learning: A Survey. [[Paper](https://semanticscholar.org/paper/c56eb865d1c9427602eecd89869eb2faf700f4f6)]

- A survey on deep learning-based fine-grained object classification and semantic segmentation. [[Paper](https://semanticscholar.org/paper/a65d672715e2c58a01d693b48b9f14b68d4916bf)]


## Papers

### 2023

- Fine-Grained Visual Classification via Internal Ensemble Learning Transformer. Xu Q / Luo B. Anhui University, CN. Transactions on Multimedia 2023. [[Paper](https://semanticscholar.org/paper/595adb75ddeb90760c79e89b76d99e55079e0708)]

  - Select intermediate tokens based on head-wise attention voting average + gaussian kernel -> multi-layer refinement, dynamic ratio of intermediate layers contributions for refinement modules

- Dual Transformer with Multi-Grained Assembly for Fine-Grained Visual Classification. Ji RY / Wu YJ. Chinese Academy of Sciences, CN. TCSVT 23. [[Paper](https://www.semanticscholar.org/paper/2fe009c6a4ae8354e20cb5f61cbbe2630a40ca47)]
  - Early crop based on 1st layer attention, attention to select tokens from intermediate features, cross-attention for interactions between CLS token of global and crops and features of other branch

- Fine-grained Classification of Solder Joints with {\alpha}-skew Jensen-Shannon Divergence. Ulger F / Gokcen D. TCPMT 23. [[Paper](https://semanticscholar.org/paper/9526a454422eb5436ee355f6e7ac94f66466ab8a)]
  - Maximize entropy to penalize overconfidence

- Shape-Aware Fine-Grained Classification of Erythroid Cells. Wang Y / Zhou Y. JLU, CN. Applied Intelligence 23. [[Paper](https://semanticscholar.org/paper/005041c7ae9debb55401314b1400b90c58f4f50d)]
  - Dataset and method for fine-grained erythroid cell classification

- Test-Time Amendment with a Coarse Classifier for Fine-Grained Classification. Jain K / Gandhi V. IIIT Hyderabad, IN. arXiv 2023/02. [[Paper](https://semanticscholar.org/paper/5cc86e3660b1fffbe347c1c8bf1613ea69b34aca)]
  - Hierarchical prediction by taking into account predictions from coarser levels (multiplication of scores)

- Semantic Feature Integration network for Fine-grained Visual Classification. Wang H / Luo HC. Jiangnan U, CN. arXiv 23/02. [[Paper](https://semanticscholar.org/paper/f25162f09360ef5782ee979b7da4732b1510fb8a)]
  - Intermediate predictions classifiers + loss (similar to SAC arXiv 22 and PIM arXiv 22) + sequence of modules to refine most discriminative intermediate features

- Learning Common Rationale to Improve Self-Supervised Representation for Fine-Grained Visual Recognition Problems. Shu YY / Hengel AVD / Liu LQ. U of Adelaide, AU. arXiv 23/03. [[Paper](https://semanticscholar.org/paper/1ba384d5a3bc52b4fc0ba53b57465851816d4fe3)]
  - Extends SAM (ECCV 22) for self-supervised setting (add GradCAM branch trained with KD loss to predict discriminative regions

- Fine-grained Visual Classification with High-temperature Refinement and Background Suppression. Chou PY / Lin CH. National Taiwan Normal U, TW. arXiv 23/03. [[Paper](https://semanticscholar.org/paper/802dbfe2dac0d7e6b3a302fd2e721bc5bc821932)]
  - Extends PIM (arXiv 22) with loss to supress background (predict -1 for background regions) + KD loss between two inter classifiers


### 2022

- MetaFormer: A Unified Meta Framework for Fine-Grained Recognition. Diao QS / Yuan Z. ByteDance, CN. arXiv 22/03. [[Paper](https://semanticscholar.org/paper/83bc100f248b25cad52b5a64ba42c771dacff437)]
  - Incorporate multimodality data as extra information (date, location, text, attributes, etc)

- Dynamic MLP for Fine-Grained Image Classification by Leveraging Geographical and Temporal Information. Yang LF / Yang J. Nanjing U of S&T, CN. CVPR 2022. [[Paper](https://semanticscholar.org/paper/484f9435d591a391f2a3a79ddc6366248227651e)]
  - Incorporate metadata (date/loc)

- Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification. Zhu HW / Shan Y. AMD, CN. CVPR 22. [[Paper](https://semanticscholar.org/paper/470ddab4aff61cb4c58d3df3966c34e7ee9ffde2)]
  - Cross-attention between selected queries and all keys/values for refinement + cross-attention for regularization (mix queries/keys/values from two images)

- SIM-Trans: Structure Information Modeling Transformer for Fine-grained Visual Categorization. Sun HB / Peng YX. Peking U, CN. ACM MM 22. [[Paper](https://semanticscholar.org/paper/ee468ee5494bb1a10e45cce05b28e63bdcf7ed40)]
  - Refine attention selected tokens using GCN & polar coordinates + contrastive loss for last 3 layers

- A Novel Plug-in Module for Fine-Grained Visual Classification. Chou PY / Kao WC. National Taiwan Normal U, TW. arXiv 22/02. [[Paper](https://semanticscholar.org/paper/b97dff4cfa5ff7ae48fc2eaea384599517e02e5f)]
  - Intermediate classifier distribution sharpness as metric to select intermediate features + GCN to combine

- ViT-NeT: Interpretable Vision Transformers with Neural Tree Decoder. Kim SW / Ko BC. Keimyung U, SK. ICML 22. [[Paper](https://www.semanticscholar.org/paper/b503f607c8e73a117888e0d5f658c6855a11c319)]
  - Binary tree with differentiable routing and refinement at each node/leaf

- Fine-Grained Object Classification via Self-Supervised Pose Alignment. Yang XH / Tian YH. Peng Cheng Lab, CN. CVPR 22. [[Paper](https://semanticscholar.org/paper/82ddbd98691dc0b3b0cc1801dab20c2f52f40400)]
  - Intermediate features classifiers with different label smoothing levels and graph matching to align parts for contrastive learning

- On the Eigenvalues of Global Covariance Pooling for Fine-grained Visual Recognition. Song Y / Wang W. U of Trento, IT. TPAMI 22. [[Paper](https://semanticscholar.org/paper/43ea18312102bbe013a1f5caf295e827b73fff28)]
  - Second order methods (B-CNN) weaknesses: small eigenvalues so propose scaling factor to magnify

- Improving Fine-Grained Visual Recognition in Low Data Regimes via Self-Boosting Attention Mechanism. Shu YY / Liu LQ. U of Adelaide, AU. ECCV 22. [[Paper](https://semanticscholar.org/paper/2312f5de420af0ed898912a76af0d9ef53369fc7)]
  - KL divergence between CAMs and convolutional projection  as auxiliary task

- SR-GNN: Spatial Relation-aware Graph Neural Network for Fine-Grained Image Categorization. Bera A / Behera A. BITS, IN / Edge Hill U, UK. TIP 22. [[Paper](https://semanticscholar.org/paper/f6bdb7b982878c38080c10b9562a865cb34b4144)]
  - Divide into regions, refinement with GNN and SA

- Cross-Part Learning for Fine-Grained Image Classification. Liu M / Zhao Y. Beijing Jiaotong University, CN. TIP 2022. [[Paper](https://www.semanticscholar.org/paper/95bc30235c42e26e476304050cc1681d70a56d33)]
  - Multi-stage processing and localization (object -> parts) + refinement

- Convolutional Fine-Grained Classification With Self-Supervised Target Relation Regularization. Liu KJ / Jia K. South China U of Technology, CN / Peng Cheng Lab, CN. arXiv 22/08. [[Paper](https://semanticscholar.org/paper/f69f0d4a1738bb3d8e011bb2557599a1d0e3fe64)]
  - Class center + distance between graphs as self-supervised loss

- R2-Trans: Fine-Grained Visual Categorization with Redundancy Reduction. Wang Y / You XG. Huazhong U, CN. arXiv 22/04. [[Paper](https://semanticscholar.org/paper/f7a6135977d9fd28d19d82aec45e68706004fe6d)]
  - Mask tokens based on attention + information theory inspired loss

- Knowledge Mining with Scene Text for Fine-Grained Recognition. Wang H / Liu WY. Huazhong U of Science and Technology, CN / Tencent, CN. CVPR 22. [[Paper](https://semanticscholar.org/paper/44b4abcaacab279c59c32a0943b49b25579bc862)]
  - Incorporate wikipedia knowledge from scene text as additional data

- Fine-Grained Visual Classification using Self Assessment Classifier. Do T / Nguyen A. AIOZ, SN / U of Liverpool, UK. arXiv 22/05. [[Paper](https://semanticscholar.org/paper/ffc96e7873c4167c6c204e7b3a6d49dfafb47f30)]
  - Predict once, augment top-k predictions with class text names to predict again

- Exploiting Web Images for Fine-Grained Visual Recognition via Dynamic Loss Correction and Global Sample Selection. Liu HF / Xiu WS / Tang ZM. Nanjing U of S&T, CN. TMM 2022. [[Paper](https://www.semanticscholar.org/paper/d523b38846a4dedeb0cdfbdacbbde8792d754c7c)]
  - Web images for fine-grained recognition

- Cross-layer Attention Network for Fine-grained Visual Categorization. Huang RR / Yang HZ. Tsinghua U, CN. arXiv 22/10 / CVPR 22 FGVC8 Workshop. [[Paper](https://semanticscholar.org/paper/828c7971b3118718bc3e80e116ded4da0e6cdcb9)]
  - Refine intermediate features with top-level and top-level with intermediate features

- Anime Character Recognition using Intermediates Feature Aggregation. Rios EA / Lai BC. National Yang Ming Chiao Tung U, TW. ISCAS 22. [[Paper](https://www.semanticscholar.org/paper/17a3c8e25db4df379a77d5e569e19d757ce2a2b1)]
  - Concatenate ViT intermediate CLS tokens and forward through fully connected layer to aggregate intermediate features + incorporate tag information as additional data. 

- Fine-grained visual classification with multi-scale features based on self-supervised attention filtering mechanism. Chen H / Ling W.  Guangdong U of T, CN. Applied Intelligence 2022. [[Paper](https://semanticscholar.org/paper/eb6759a43c1f91704d74b103b4e63e592b98e191)]
  - Attention map filtering and multi-scale

- Bridge the Gap between Supervised and Unsupervised Learning for Fine-Grained Classification. Wang JB / Wei XS / Zhang R.  Army Engineering U of PLA, CN / Nanjing U, CN.  arXiv 22/03. [[Paper](https://semanticscholar.org/paper/609cbfa75cb8698786bc3b1124e17708a656d9a9)]
  - Study on unsupervised fine-grained (no labels, clustering-based)

- PEDTrans: A fine-grained visual classification model for self-attention patch enhancement and dropout. Lin XH / Chen YF. China Agricultural U, CN. ACCV 22. [[Paper](https://www.semanticscholar.org/paper/3ddbb24e8dd635e5ffae717c537cb18d8d615c78)]
  - Patch dropping based on similarity (outer product/bilinear pooling) + refinement of patches before transformer

- Iterative Self Knowledge Distillation -- from Pothole Classification to Fine-Grained and Covid Recognition. Peng KC. Mitsubishi MERL, US. ICASSP 22. [[Paper](https://semanticscholar.org/paper/26bc39789c38fc0aeadfae85b0aa28404f3e4ec4)]
  - Use student from previous iteration as teacher, recursively

- Fine-grain Inference on Out-of-Distribution Data with Hierarchical Classification. Linderman R / Chen Y. Duke U, US. NeurIPS 22 Workshop. [[Paper](https://semanticscholar.org/paper/22caecd237f83dfa635ba4f33ce1d7e71ae979a5)]
  - Hierarchical OOD fine-grained with inference stopping criterion

- Semantic Guided Level-Category Hybrid Prediction Network for Hierarchical Image Classification. Wang P / Qian YT. Zhejiang University, CN. arXiv 2022/11. [[Paper](https://semanticscholar.org/paper/cad01259edc56cdc0086f012cf6fab3f517aa502)]
  - Hierarchical prediction taking into account “quality” (noise, occlusion, blur or low resolution) to decide classification level

- Data Augmentation Vision Transformer for Fine-grained Image Classification. Hu C / Wu WJ. Unknown affiliation. arXiv 22/11. [[Paper](https://semanticscholar.org/paper/ad038a775d2bb7c9f3412c35d81a692aaaa14616)]
  - Crops based on single-layer (5th) attention + TransFG’s PSM module between 2 layers (recursive matrix-matrix attention)

- Medical applications (COVID, kidney pathology, renal and ocular disease):
  - Self-supervision and Multi-task Learning: Challenges in Fine-Grained COVID-19 Multi-class Classification from Chest X-rays. Ridzuan M / Yaqub M. MBZUAI, AE. MIUA 22. [[Paper](https://semanticscholar.org/paper/b48f7a3ba1c3c16e0b70dc867182e549ca90be94)]

  - Automatic Fine-grained Glomerular Lesion Recognition in Kidney Pathology. Nan Y / Yang G. Imperial College London, UK. Pattern Recognition 22. [[Paper](https://semanticscholar.org/paper/9beb399bddc399efea07f4d3d0cb36e5236f71fc)]

  - Holistic Fine-grained GGS Characterization: From Detection to Unbalanced Classification. Lu YZ / Huo YK. Vanderbilt U, US. Journal Medical Imaging 2022. [[Paper](https://semanticscholar.org/paper/e92e0c34d9514279125a3a85230e5c8b98589277)]

  - CDNet: Contrastive Disentangled Network for Fine-Grained Image Categorization of Ocular B-Scan Ultrasound. Dan RL / Wang YQ. Hangzhou Dianzi U, CN. arXiv 22/06. [[Paper](https://semanticscholar.org/paper/efd08ecf69a2dd36432ba8aaf5f0e16b3e98e6da)]

- Snake competition methodologies:
  - Solutions for Fine-grained and Long-tailed Snake Species Recognition in SnakeCLEF 2022. Zou C / Cheng Y. Ant Group, CN. Conference and Labs of the Evaluation Forum 2022. [[Paper](https://semanticscholar.org/paper/80943bd4586999494611ec50bcfaf8d937d2c385)]

  - Explored An Effective Methodology for Fine-Grained Snake Recognition. Huang Y / Feng JH. Huazhong U of Science and T, CN / Alibaba, CN. CLEF 22. [[Paper](https://semanticscholar.org/paper/7af27a6f089f4538fcc68b1b2f138b991f457ffd)]


### 2021
- First ViTs for FGIR:
    - TransFG: A Transformer Architecture for Fine-Grained Recognition. He J / Wang CH. Johns Hopkins U / ByteDance. arXiv 21/03 / AAAI 22. [[Paper](https://semanticscholar.org/paper/860e24025c67487b9dd87b442c7b44e5bbf5a054)]
      - First to apply ViT for FGIR: overlapping patchifier convolution, recursive layer-wise matrix-matrix multiplication to aggregate attention and select features from last layer, contrastive loss

    - Feature Fusion Vision Transformer for Fine-Grained Visual Categorization. Wang J / Gao YS. U of Warwick, UK / Griffith U, AU. BMVC 21. [[Paper](https://semanticscholar.org/paper/64d8af9153d68e9b50f616d227663385bece93b9)]
      - ViT for FGIR, select intermediate tokens based on layer-wise attention

    - RAMS-Trans: Recurrent Attention Multi-scale Transformer for Fine-grained Image Recognition. Hu YQ / Xue H. Zhejiang U / Alibaba, CN. ACM MM 21. [[Paper](https://semanticscholar.org/paper/21288897fd9e8ce56104743138caffa00470ca13)]
      - ViT for FGIR, select regions to crop based on recursive layer-wise attention matrix-matrix multiplication + individual CLS token for crops

    - Transformer with peak suppression and knowledge guidance for fine-grained image recognition. Liu XD / Han XG. Beihang U, CN. Neurocomputing 22. [[Paper](https://semanticscholar.org/paper/2fa6c8396d214c4063a9acd2068e639f384e9f2c)]
      - ViT for FGIR, mask tokens of top attention to prevent overconfident predictions, learnable class matrix to augment output 

    - A free lunch from ViT: adaptive attention multi-scale fusion Transformer for fine-grained visual recognition. Zhang Y / Chen WQ. Peking U / Alibaba, CN. arXiv 21/08 ICASSP 22. [[Paper](https://semanticscholar.org/paper/401c8d72a9b275e88e6ba159d8d646cfb9f397aa)]
      - ViT for FGIR, crops based on head-wise element-wise multiplications of attention heads and aggregating through SE-like mechanism to reweight different layers attentions

    - Exploring Vision Transformers for Fine-grained Classification. Conde MV / Turgutlu K. U of Valladolid, ES. CVPR Workshop 21. [[Paper](https://semanticscholar.org/paper/bd13e8695f0b59829c3cd0d81456fb7f678970b6)]
      - ViT for FGIR, attention rollout + morphological operations for recursive cropping / masking

    - Complemental Attention Multi-Feature Fusion Network for Fine-Grained Classification. Miao Z / Li H. Army Eng U of PLA, CN. Signal Proc Letters 21. [[Paper](https://semanticscholar.org/paper/fb3b47814d344e9e19760c4bbb8686c6664c2e01)]
      - Reweight Swin features based on importance and divide into two branches (discriminative and not) 

    - Part-Guided Relational Transformers for Fine-Grained Visual Recognition. Zhao YF / Tian YH. Beihang U, CN. TIP 21. [[Paper](https://semanticscholar.org/paper/160efdab15386cc890b436fb79c536bdecf228d3)]
      - Transformer with positional embeddings from CNN features to refine global and part features

    - A Multi-Stage Vision Transformer for Fine-grained Image Classification. Huang Z / Zhang HB. Huaqiao U, CN. ITME 21. [[Paper](https://www.semanticscholar.org/paper/b525c44228e09253c5d51d4846bfe61b38c65fb2)]
      - ViT for FGIR with pooling layer to build multiple stages in transformer

- AP-CNN: Weakly Supervised Attention Pyramid Convolutional Neural Network for Fine-Grained Visual Classification. Ding YF / Ma ZY / Ling HB. Beijing U of Posts & Telecomms, CN. TIP 21. [[Paper](https://www.semanticscholar.org/paper/7d8495a37bf6473901334b0691336eb1e5b50d87)]
  - FPN with top-down & bottom-up paths + merged ROI cropping + ROI masking

- Counterfactual Attention Learning for Fine-Grained Visual Categorization and Re-identification. Rao YM / Zhou J. Tsinghua U, CN. ICCV 21. [[Paper](https://semanticscholar.org/paper/0f64bf963ab8c5fe341c4cb9679d18e4d57917ec)]
  - Builds on WS-DAN (attention crop & mask) by making predictions with counterfactual (fake) attention maps to learn better attention maps

- Neural Prototype Trees for Interpretable Fine-grained Image Recognition. Nauta M/ Seifert C. University of Twente, NL. CVPR 21. [[Paper](https://www.semanticscholar.org/paper/53b224126e7abe03e21186f47fcf9681e1ef5909)]
  - Binary trees based on similarity to protoypes + pruning

- SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data. Huang SL / Tao DC. U of Sydney, AU. AAAI 21. [[Paper](https://www.semanticscholar.org/paper/c3379bbe58886d8d66c0be777fc7416415617435)]
  - CutMix (cut part fron one image into another as data aug) with asymmetric crops + assign labels based on CAMs

- Intra-class Part Swapping for Fine-Grained Image Classification. Zhang LB / Huang SL / Liu W. U of Technology Sydney, AU. WACV 2021. [[Paper](https://www.semanticscholar.org/paper/75fd71f9fcf03fefb5f8a535db68bc40ccd2a8e2)]
  - CutMix images from same class only + affine transform guided by CAMs for mixing

- Stochastic Partial Swap: Enhanced Model Generalization and Interpretability for Fine-grained Recognition. Huang SL / Tao DC. The University of Sydney, AU. ICCV 21. [[Paper](https://semanticscholar.org/paper/89a0cd43aaac11cfcd0ed127dfdd55fe59527be6)]
  - Intermediate classifiers + changing features of one image with another randomly to inject noise

- Enhancing Mixture-of-Experts by Leveraging Attention for Fine-Grained Recognition. Zhang LB / Huang SL / Liu Wei. U of Technology Sydney / U of Sydney, AU. TMM 21. [[Paper](https://www.semanticscholar.org/paper/284b465ddcc2b2ef5d2d92f2de23906533c399a9)]
  - CutMix based on activations from last conv layer, same class only, crops also based on activations from last conv

- Multiresolution Discriminative Mixup Network for Fine-Grained Visual Categorization. Xu KR / Li YS. Xidian U, CN. TNNLS 21. [[Paper](https://semanticscholar.org/paper/85b98a0704ef45d04fadb067f4d94d0c9c666415)]
  - Mixup based on CAM attention + distillation from multiple high resolution crops to single low resolution crop

- Context-aware Attentional Pooling (CAP) for Fine-grained Visual Classification. Behera A / Bera A. Edge Hill U, UK. AAAI 21. [[Paper](https://semanticscholar.org/paper/48913aecd8da6475d93fae7beb13d7ef939b3d8a)]
  - Combine cross-regions features with attention + LSTM + learnable pooling

- A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification. Su JC / Maji S. U of Massachusetts Amherst, US. CVPR 21. [[Paper](https://semanticscholar.org/paper/7ce8f0dda13a434314562f92d56147c7970f1c62)]
  - In depth-study on fine-grained semi-supervised learning

- MaskCOV: A random mask covariance network for ultra-fine-grained visual categorization. Yu XH / Xiong SW. Griffith U, AU / Wuhan U of T, CN. Pattern Recognition 21. [[Paper](https://semanticscholar.org/paper/00acec4d0e490818c510cac1994fbfc1dcbab29a)]
  - Masking and shuffling of patches as data aug, predict covariance as auxiliary task

- Benchmark Platform for Ultra-Fine-Grained Visual Categorization Beyond Human Performance. Yu XQ / Xiong SW. Griffith U, AU / Wuhan U of T, CN. ICCV 21. [[Paper](https://semanticscholar.org/paper/f328f8746e0e32fb167028a71c114b250fa7be89)]
  - Ultra fine-grained recognition of leaves dataset

- Human Attention in Fine-grained Classification. Rong Y / Kasneci E. University of Tübingen, DE. BMVC 21. [[Paper](https://semanticscholar.org/paper/ccd1b95f992db9349e4bc7e1e9ebdb886301d634)]
  - Human attention/gaze for crops/extra modality data

- Fair Comparison: Quantifying Variance in Results for Fine-grained Visual Categorization. Gwilliam M / Farrell R. Brigham Young U, US / U of Maryland, US. WACV 21. [[Paper](https://semanticscholar.org/paper/c6995f98882a201a51d6374722e39dc5f97f4327)]
  - Study on the failure of single top-1 accuracy as metric for FGIR, suggest using class variance and standard deviation and mean of multiple experiments with different random seeds

- Learning Canonical 3D Object Representation for Fine-Grained Recognition. Joung SH / Sohn KH. Yonsei U, KR. ICCV 21. [[Paper](https://www.semanticscholar.org/paper/1e24118e9822dfa5d82a2b3c4b3b6738206b8bd2)]
  - Learn 3D representations as auxiliary task for fine grained recognition

- Multi-branch and Multi-scale Attention Learning for Fine-Grained Visual Categorization. Zhang F / Liu YZ. China U of Mining and T, CN. MMM 21. [[Paper](https://www.semanticscholar.org/paper/023379daf1e4168273e460768367bb725027a198)]
  - Features maps of multiple layers (instead of one) to guide cropping

- CLIP-Art: Contrastive Pre-training for Fine-Grained Art Classification. Conde MV / Turgutlu K. U of Valladolid, ES. CVPR Workshop 21. [[Paper](https://semanticscholar.org/paper/dc3c161fee18dfe879279bbf931f0b5f5176c3d8)]
  - Applies CLIP for fine-grained art recognition

- Graph-based High-Order Relation Discovery for Fine-grained Recognition. Zhao YF /  Li J. Beihang University, CN. CVPR 21. [Paper](https://semanticscholar.org/paper/c4827bea40963c5fb57754a80d58235d7e78b288)]
  - Extend on bi/trilinear pooling + GCN for refining features

- Progressive Learning of Category-Consistent Multi-Granularity Features for Fine-Grained Visual Classification. Du RY / Ma ZY / Guo J. Beijing U of Posts and Telecomms, CN. TPAMI 21. [[Paper](https://semanticscholar.org/paper/d28c92312c996d67cdb080d829acac7a65389623)]
  - Extended journal version of PMG (ECCV20): progressive training with block-based processing + pair category consistency loss between same class images

- Webly Supervised Fine-Grained Recognition: Benchmark Datasets and An Approach. Sun ZR / Wei XS / Shen HT. Nanjing U of S&T / Nanjing U, CN. ICCV 21.  [[Paper](https://semanticscholar.org/paper/d7795a25aba9c5714b4ac748d62534d5dd624b02)]
  - Dataset for fine-grained recognition with noisy web labels and method to train with noisy labels

- Re-rank Coarse Classification with Local Region Enhanced Features for Fine-Grained Image Recognition. Yang SK / Liu S / Wang CH ByteDance, CN. arXiv 21/02. [[Paper](https://semanticscholar.org/paper/4bd8f42cc06f64304bd4655d2ae738272af76252)]
  - Automatic hierarchy based on clustering, triplet loss to guide crops, similarity to class database to re-classify images (compared to coarse classifier)

- Progressive Co-Attention Network for Fine-grained Visual Classification. Zhang T / Ma ZY / Guo J. Beijing U of Posts and Telecomms, CN. VCIP 21. [[Paper](https://www.semanticscholar.org/paper/fcd3abb24f07b45525044efd8107eb3e530ff092)]
  - Interaction between pairs of images using bilinear pooling

- Subtler mixed attention network on fine-grained image classification. Liu C / Zhang WF. Ocean U of China, CN. Applied Intelligence 21. [[Paper](https://semanticscholar.org/paper/7506eed7e40d84dff829e1574015566acaaa6aa2)]
  - Spatial and channel attention on parts

- Dynamic Position-aware Network for Fine-grained Image Recognition. Wang SJ / Li HJ / Ouyang WL. Dalian U of T, CN. AAAI 21. [[Paper](https://semanticscholar.org/paper/cf0723eab4c75e5b6bdfeca38aa80e76531aaa1b)]
  - Horizontal and vertical pooling + learnable sin/cos positional embeddings + GCN for crops

- Learning Scale-Consistent Attention Part Network for Fine-Grained Image Recognition. Liu HB / Lin WY. Shanghai Jiaotong U, CN. TMM 21. [[Paper](https://semanticscholar.org/paper/c7b2e98390bf634ca50cee9e5d3a9319cb3b127d)]
  - SE-like + Gumbel softmax trick + scale-consistency for parts detection + self-attention for parts relations

- Multi-branch Channel-wise Enhancement Network for Fine-grained Visual Recognition. Li GJ / Zhu FT. University of Shanghai for Science and Technology, CN. ACM MM 21. [[Paper](https://semanticscholar.org/paper/41c638736c8693eb11132f9281659b305096827b)]
  - Multi-size spatial shuffling (similar to DCL (CVPR19) but with multiple sizes of shuffling)

- Fine-Grained Categorization From RGB-D Images. Tan YH / Lu K. Chinese Academy of Sciences, CN. TMM 21. [[Paper](https://www.semanticscholar.org/paper/c37ccd28ad4abf58a6a15f97a24983925874123a)]
  - Dataset and network for incorporating RGB and depth images


### 2020


- The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification. Chang DL / Song YZ. U of Posts and Telecomms, CN. TIP 20. [[Paper](https://semanticscholar.org/paper/3f07093e303a8a9ddc157c2fceb47802bb9b5613)]
  - Channel groups loss to make each channel group discriminative and focus on different spatial regions

- Learning Attentive Pairwise Interaction for Fine-Grained Classification. Zhuang PQ / Qiao Y. Chinese Acad. Of Sciences, CN. AAAI 20. [[Paper](https://semanticscholar.org/paper/e3ce51203b7763c9f3f20a62045578e562906d52)]
  - Pairwise interactions between pairs of images from same/different class

- Fine-Grained Visual Classification via Progressive Multi-Granularity Training of Jigsaw Patches. Du RY / Guo J. U of Posts and Telecomms, CN. ECCV 20. [[Paper](https://semanticscholar.org/paper/293ed5672bc3f86051af38ad58b4e09fe2d1aa92)]
  - Jigsaw puzzle for data augmentation of different network stages, training each stage progressively and classifier for each stage

- Channel Interaction Networks for Fine-Grained Image Categorization. Gao Y / Scott M. Malong Technologies, CN. AAAI 20. [[Paper](https://semanticscholar.org/paper/8585737f242285ad322acf23bc4943ddd50bf045)]
  - Trilinear pooling + contrastive loss to pull images from same class together and push images from different class apart

- ELoPE: Fine-Grained Visual Classification with Efficient Localization, Pooling and Embedding. Hanselmann H / Ney H. WTH Aachen U, DE. WACV 20. [[Paper](https://www.semanticscholar.org/paper/2e6949c9293156a105e64d9910a77a2160a0bca0)]
  - Small CNN to predict crops + embedding loss w/ class centers

- Fine-Grained Visual Classification with Efficient End-to-end Localization.  Hanselmann H / Ney H. arXiv 20/05. [[Paper](https://www.semanticscholar.org/paper/424928376b6ea2b6a5509d61f31ec6309ce85cfb)]
  - End-to-end train of small CNN + STN

- Attentional Kernel Encoding Networks for Fine-Grained Visual Categorization. Hu YT / Zhen XT. Beihang U, CN. TCSVT 20. [[Paper](https://semanticscholar.org/paper/f174b25d2c41bf9a436252c322c8b76712cd0b43)]
  - Cascaded attention + fourier/cosine kernel (cos of input)

- Bi-Modal Progressive Mask Attention for Fine-Grained Recognition. Song KT / Wei XS / Lu JF. Nanjing U of S&T, CN. TIP 20. [[Paper](https://semanticscholar.org/paper/5820780c77bc53854eb336e1d748c9390f02b40e)]
  - Multi-stage fusion of vision (CNN) & text (LSTM) with vision-/language-only attention & cross-modality attention and intermediate classifiers

- Hierarchical Image Classification using Entailment Cone Embeddings. Dhall A / Krause A. ETH Zurich, CH. CVPR Workshop 20. [[Paper](https://semanticscholar.org/paper/a282a72322d6d0d39a9ca20bc68701c65d79fd4e)]
  - Comparison on losses and embeddings for hierarchical classification

- Learning Semantically Enhanced Feature for Fine-Grained Image Classification. Luo W / Wei XS. IEEE, US. Signal Processing Letters 20. [[Paper](https://semanticscholar.org/paper/7ada0aca1c2b3213de6610a9d2ffd76554ff9d79)]
  - Group feature channels based on semantics and KD from global features to groups

- An Adversarial Domain Adaptation Network For Cross-Domain Fine-Grained Recognition. Wang YM / Wei XS / Zhang LJ. Nanjing U, CN / Megvii. WACV 20. [[Paper](https://semanticscholar.org/paper/54cc2e745cce362d3501d75d603801428f939767)]
  - Adversarial loss to distinguish domains + loss to pull features from same class together + attention binary mask for removing BG

- Group Based Deep Shared Feature Learning for Fine-grained Image Classification. Li XL /  Monga V. Pennsylvania State University, US. BMVC 20. [[Paper](https://semanticscholar.org/paper/365bb5a062b3ea36eb6b66454eb4d847d688c2b0)]
  - Autoencoder with class/shared center loss to divide features into class and not

- Beyond the Attention: Distinguish the Discriminative and Confusable Features For Fine-grained Image Classification. Shi XR / Liu W. Beijing U of Posts and Telecomm, CN. ACM MM 20. [[Paper](https://semanticscholar.org/paper/5295d66f189a092609323bc07491a7702596d38f)]
  - Divide into discriminative/confusing regions w/ SE to refine features, intermediate losses for classification, pulling features of images with same label closer (L1) and maximizing entropy of confusing features (pseudolabel of 1 to all classes -> background)

- Fine-Grained Classification via Categorical Memory Networks. Deng WJ / Zheng L. Australian National U, AU. arXiv 20/12 / TIP 22. [[Paper](https://semanticscholar.org/paper/784ddb9486df61d2b171a94a35582c60ae9404ca)]
  - Augment feature with class-specific memory module (learned average based on previous samples and how similar / how it reacts to new samples)

- Interpretable and Accurate Fine-grained Recognition via Region Grouping. Huang ZX / Li Y. U of Wisconsin-Madison, US. CVPR 20. [[Paper](https://semanticscholar.org/paper/f4cc061e63a88ae9ae3192ae6985b1699287f54d)]
  - Part assignment, feature refinement and weighted classification

- Filtration and Distillation: Enhancing Region Attention for Fine-Grained Visual Categorization. Liu CB / Zhang YD. U of S&T of China, CN. AAAI 20. [[Paper](https://semanticscholar.org/paper/a429a700ceca9ef94be68a9652e76e2870fa6803)]
  - RPN with losses for consistency between proposals from RPN and main feature extractor + KD between object and parts

- Graph-Propagation Based Correlation Learning for Weakly Supervised Fine-Grained Image Classification. Wang ZH / Li HJ / Li JJ. Dalian U of S&T, CN. AAAI 20. [[Paper](https://semanticscholar.org/paper/55dad10417658173dd92b83f0ab2069d60f2b46e)]
  - GCN for graph propagation for discriminative feature selection (crops) + losses for cropping

- Weakly Supervised Fine-grained Image Classification via Gaussian Mixture Model Oriented Discriminative Learning. Wang ZH / Li HJ / Li ZZ. Dalian U of T, CN. CVPR 20. [[Paper](https://semanticscholar.org/paper/aa7580bf7a8e282b754837bacfa91d9773358b10)]
  - Gaussian mixture model to learn low rank feature maps for selecting crops

- Category-specific Semantic Coherency Learning for Fine-grained Image Recognition. Wang SJ / Li HJ / Ouyang WL. Dalian U of T, CN. ACM MM 20. [[Paper](https://semanticscholar.org/paper/fb3437942a89a1260db859620127292516eee67d)]
  - Latent attributes prediction, alignment, reordering and patch-wise attention for selecting crops

- Multi-Objective Matrix Normalization for Fine-Grained Visual Recognition. Min SB / Zhang YD. U of S&T of China, CN. TIP 20. [[Paper](https://semanticscholar.org/paper/22970ea2363642841e7fb58e20150df4b1b8bfbc)]
  - Matrix normalization for bilinear pooling

- Power Normalizations in Fine-Grained Image, Few-Shot Image and Graph Classification. Koniusz P / Zhang HG. Australian National U, AU. TPAMI 20. [[Paper](https://semanticscholar.org/paper/70a9706d37811508f5d0b27c435404f64e8c2ad5)]
  - Study on normalizations for B-CNN

- Fine-grained Image Classification and Retrieval by Combining Visual and Locally Pooled Textual Features. Mafla A / Karatzas D. UAB, ES. WACV 20. [[Paper](https://semanticscholar.org/paper/871f316cb02dcc4327adbbb363e8925d6f05e1d0)]
  - Extract and incorporate text in images for FGIR

- Multi-Modal Reasoning Graph for Scene-Text Based Fine-Grained Image Classification and Retrieval. Mafla A / Karatzas D. UAB, ES. arXiv 20/09 / WACV 21. [[Paper](https://semanticscholar.org/paper/b67759f193e2c39877723424df0b3d5f91c0bf0b)]
  - Expands on previous by encoding multimodality with GCN

- Focus Longer to See Better: Recursively Refined Attention for Fine-Grained Image Classification. Shroff P / Wang ZY. Texas A&M U, US. CVPR Workshop 20. [[Paper](https://semanticscholar.org/paper/15e2d3d9b4ea425295478a7f379e6f45633cdd5d)]
  - Recursive LSTM for encoding cropped features

- Fine-Grained Visual Categorization by Localizing Object Parts With Single Image. Zheng XT / Lu XQ. Chinese Acad of Sciences, CN. TMM 20. [[Paper](https://semanticscholar.org/paper/1aa5f1f5496fea505f5dba4a8d0a2af3edfc670f)]
  - Cluster feature maps of multiple layers

- Microscopic Fine-Grained Instance Classification Through Deep Attention. Fan MR / Rittscher J. U of Oxford, UK. MICCAI 20. [[Paper](https://semanticscholar.org/paper/e7685f05efccce4f54e2fe24a320e96fedc64390)]
  - Attention crops for microscopic applications


### 2019


- Destruction and Construction Learning for Fine-Grained Image Recognition. Chen Y / Mei T. JD AI Research, CN. CVPR 19. [[Paper](https://semanticscholar.org/paper/c947c89c4a709ad27fe4590294589196642b0214)]
  - Shuffling local regions in an image (destruction) + learning to predict original locations (construction) + adversarial loss to distinguish shuffled from not 

- Looking for the Devil in the Details: Learning Trilinear Attention Sampling Network for Fine-Grained Image Recognition. Zheng HL /  Luo JB. U of S&T of CN, CN. CVPR 19. [[Paper](https://semanticscholar.org/paper/9a9959260309cd7acfa6b582a5deb33168e0531a)]
  - Trilinear attention (〖𝑿𝑿〗^𝑻 𝑿) for crops + KD loss between crops & original

- Weakly Supervised Complementary Parts Models for Fine-Grained Image Classification From the Bottom Up. Ge WF / Yu YZ. U of Hong Kong, HK. CVPR 19. [[Paper](https://semanticscholar.org/paper/cf9d7f064a702f165ad8b24b41898cb3a74eb24f)]
  - Weakly supervised detection/segmentation with Mask R-CNN, CAMs & CRFs + LSTM for aggregating features from original and crops

- See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification. Hu T / Lu Y. Chinese Academy of Sciences, CN / Microsoft. arXiv 19. [[Paper](https://semanticscholar.org/paper/a21d8d27caca866d8f107116a2c90a2ed24b0b7a)]
  - Attention masking (and cropping) + moving average center loss to guide attention maps

- Selective Sparse Sampling for Fine-grained Image Recognition. Ding Y / Jiao JB. U of Chinese Academy of Sciences, CN. ICCV 19. [[Paper](https://semanticscholar.org/paper/a50e6e69a62ca07cd5fbe7befb96af81bedf71bb)]
  - CAMs peaks + Gaussians based on classification entropy (confidence) for resampling images (cropping with convs)

- Cross-X Learning for Fine-Grained Visual Categorization. Luo W / Lim S. South China Agricultural University, CN / FB. ICCV 19. [[Paper](https://semanticscholar.org/paper/f47ed2a9fc6772437bb951284852cafd15bc4124)]
  - Multiple excitations (OSME with loss to distinguish excitations) with intermediate features (FPN + KD loss between intermediate predictions)

- P-CNN: Part-Based Convolutional Neural Networks for Fine-Grained Visual Categorization. Han JW / Xu D. Northwestern Polythechnical U, CN / U of Sydney, AU. TPAMI 19. [[Paper](https://semanticscholar.org/paper/684666f4d2e529411f73616a166a45098cddf1bb)]
  - Cluster peak channel responses using K-means as part detectors

- Learning Rich Part Hierarchies With Progressive Attention Networks for Fine-Grained Image Recognition. Zheng HL / Luo JB / Mei T. Microsoft, CN. TIP 19. [[Paper](https://semanticscholar.org/paper/3f21a925310818e45f66ac5e96649191bf559908)]
  - Journal MA-CNN w/ refinement module and iterative training

- Bidirectional Attention-Recognition Model for Fine-Grained Object Classification. Liu CB / Zhang YD. U of S&T of China, CN. TMM 19. [[Paper](https://semanticscholar.org/paper/ee5f5113e11d46fb7deb7055e159e59348114f42)]
  - RPN for proposals with feedback (NTS-Net like) + multiple random erasing data augmentation

- Deep Fuzzy Tree for Large-Scale Hierarchical Visual Classification. Wang Y / Li XQ. Tianjin U, CN. Trans. Fuzzy Systems 19. [[Paper](https://semanticscholar.org/paper/f0a17bc5d21c879b33b4260ac660c75cbe48b94e)]
  - Fuzzy tree Based on interclass similarity for hierarchical class

- Part-Aware Fine-grained Object Categorization using Weakly Supervised Part Detection Network. Zhang YB / Wang ZX. South China U of Technology, CN. TMM 19. [[Paper](https://semanticscholar.org/paper/faf4d0e92691c94faeaa88d58849d5673221a532)]
  - RPN proposals based on channel-wise peaks + self-supervised part labeling


### 2018

- Learning to Navigate for Fine-grained Classification. Yang Z / Wang LW. Peking U, CN. ECCV 2018. [[Paper](https://semanticscholar.org/paper/472e4265895de65961b70779fdfbecafc24079ed)]
  - Feedback between networks, shared feature extractor between modules, RPN (Faster-RCNN) for part proposal

- Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning. Cui Y / Belongie S. Cornell University, US. CVPR 18. [[Paper](https://semanticscholar.org/paper/89c3355f5bc7130ae4ed090c8accc52dd885d558)]
  - Importance of resolution and strategy for long-tailed and distance to capture domain similarity between datasets for better transfer learning by training on similar sources to target

- Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition. Sun M / Ding ER. Baidu, CN. ECCV 18. [[Paper](https://semanticscholar.org/paper/ab9e438195989056f471086f9becdfb3b2f90459)]
  - Multi-excitation (squeeze-and-excitation) for feature maps + loss to pull features from same excitation closer and pushes features from different excitations away

- Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition. Yu CJ / You XG. Huazhong U of S&T, CN . ECCV 18. [[Paper](https://semanticscholar.org/paper/5d608b616efb2ae57669003b6c1067d1bb7c0b4c)]
  - Combine intermediate features by element-wise multiplications + concatenation of bilinearly pooled outputs

- Deep Attention-Based Spatially Recursive Networks for Fine-Grained Visual Recognition. Wu L / Wang Y. U of Queensland, AU. Trans. Cybernetics 18. [[Paper](https://semanticscholar.org/paper/37690b336bbe01fc3eca136dce73e31b8596e9e0)]
  - Bilinear pooling w/o sum (outer product only not matrix mult)+ FC +softmax for attention + 2D spatial LSTM with neighborhood to aggregate features

- Mask-CNN: Localizing Parts and Selecting Descriptors for Fine-Grained Image Recognition. Wei XS / Wu JX. Nanjing University, CN. arXiv 16/05 (Submitted to NIPS16) / Pattern Recognition 2018/04. [[Paper](https://semanticscholar.org/paper/c278accaeb977444d99b1426cb8bc52d29bc28dd)]
  - FCN for segmentation of parts + descriptor selection for GAP/GMP

- Maximum-Entropy Fine-Grained Classification. Dubey A / Naik N. Massachusetts Institute of Technology, US. NIPS 18. [[Paper](https://semanticscholar.org/paper/0e7a30ecb8fba8cebf43668a8648c08b47dd9f31)]
  - Prevent overconfidence with maximum-entropy loss + definition of fine-grained based on diversity

- Fine-Grained Image Classification Using Modified DCNNs Trained by Cascaded Softmax and Generalized Large-Margin Losses. Shi WW. Xian Jiaotong U, CH. TNNLS18. [[Paper](https://semanticscholar.org/paper/d2c0daf189722ff66031878d41dbaaef81a48f8a)]
  - Multi objective classification with cascaded FC classifiers for each hierarchy level + loss to bring same fine-grained class together and same coarse class closer than different coarse


### 2017

- Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition. Fu JF / Zheng HL / Mei T. Microsoft / U of S&T of China, CN. CVPR 17. [[Paper](https://semanticscholar.org/paper/a0ac9d4b0b02f6eb3a5188624e87e63e5eae6709)]
  - Recurrent CNN with intra-scale classification loss and inter-scale pairwise ranking loss to enforce finer-scale to generate more confident predictions

- Learning Multi-Attention Convolutional Neural Network for Fine-Grained Image Recognition. Zheng HL / Mei T / Luo JB. U of S&T of China, CN / Microsoft. ICCV 17. [[Paper](https://semanticscholar.org/paper/8396c9a186882457e1d4c822f2ed401937674bb5)]
  - Channel grouping module to select multiple parts from CNN feature maps + loss for compact distribution and diversity with geometric constraints

- Object-Part Attention Model for Fine-Grained Image Classification. Peng YX / Zhao JJ. Peking U, CN. arXiv 17/04 / TIP 18. [[Paper](https://semanticscholar.org/paper/67b9b6db06fa91145bed02438aab8773cc029f1c)]
  - Propose automatic object localization via saliency extraction (CAM) for localizing objects, object-part spatial constraints and clustering of parts based on clustered intermediate CNN filters

- Low-Rank Bilinear Pooling for Fine-Grained Classification. Kong S / Fowlkes C. University of California Irvine, US. CVPR 17. [[Paper](https://semanticscholar.org/paper/f24bbac95cc735c42559d53a33eaf5d689e316c1)]
  - Bilinear pooling with low-dimensionality projection (extra FC layer)
- Pairwise Confusion for Fine-Grained Visual Classification. Dubey A / Naik N. MIT, US. arXiv 17/05 / ECCV 18. [[Paper](https://semanticscholar.org/paper/20cc4bfdb648fd7947c71252589fc867d4d16933)]
  - Euclidean Distance loss which “confuses” network by adding a regularization term which minimizes distance between two images in mini-batch

- Bilinear Convolutional Neural Networks for Fine-Grained Visual Recognition. Lin T / Maji S. University of Massachusetts Amherst, US. TPAMI 2017. 177. [[Paper](https://semanticscholar.org/paper/ddef0c93cd9f604e5905b81dc56818a477f171e2)]
  - Extension and analysis of bilinear pooling

- Fine-grained Image Classification via Combining Vision and Language. He XT / Peng YX. Peking U, CN. CVPR 17. [[Paper](https://semanticscholar.org/paper/dedbf1b9da95efeb198f23fad5e395ed8a349af7)]
  - Vision (GoogleNet) & language (CNN-RNN) two-stream network

- Higher-Order Integration of Hierarchical Convolutional Activations for Fine-Grained Visual Categorization. Cai SJ / Zhang L. HK Polytechnic University, HK. ICCV 17. [[Paper](https://semanticscholar.org/paper/d3cc2082f888268c9a12bbe9d973494b42ba7f8e)]
  - Bilinear pooling for multiple layers using 1x1 Convs and concatenating intermediate outputs

- The Devil is in the Tails: Fine-grained Classification in the Wild. Horn GV / Perona P. Caltech, US. ArXiv 2017/09. [[Paper](https://semanticscholar.org/paper/e52960354739d76ffcaccf3c46315f83deddc138)]
  - Discussion on challenges related to long-tailed fine-grained classification

- BoxCars: Improving Fine-Grained Recognition of Vehicles using 3D Bounding Boxes in Traffic Surveillance. Sochor J / Herout A. Brno U of T, CZ. Transactions on ITS 17. [[Paper](https://semanticscholar.org/paper/952ad66354edf09132ee37cf9ba1286c6881ce84)]
  - Automatic 3D BBox estimation for car recognition


### 2016

- Diversified Visual Attention Networks for Fine-Grained Object Classification. Zhao B / Yan SC. Southwest Jiaotong U, CN. arXiv 16/06 / TMM 17. [[Paper](https://semanticscholar.org/paper/ca277627960c2cbcd06fbbb31a7d50e5d3eb4739)]
  - Multi-scale canvas for CNN extractor + LSTM to refine CNN predictions across time steps

- Learning a Discriminative Filter Bank Within a CNN for Fine-Grained Recognition. Wang YM / Davis LS. University of Maryland, US. arXiv 16/11 / CVPR 18. [[Paper](https://semanticscholar.org/paper/32a40b045e665db39e120c12338f9f1238b0690b)]
  - Two stream head: global (original) and part with 1x1 Conv, spatial global max pooling, and filter grouping/pooling to focus on most discriminative parts

- Picking Deep Filter Responses for Fine-grained Image Recognition. Zhang XP / Tian Q. Shanghai Jiao Tong U, CN. CVPR 16. [[Paper](https://semanticscholar.org/paper/3f6a017556bcd6167526ddb97e5e77ec27f0f9b4)]
  - Selecting deep filters which react to parts + spatial-weighting of Fisher Vector

- BoxCars: 3D Boxes as CNN Input for Improved Fine-Grained Vehicle Recognition. Sochor J / Havel J. Brno U of T, CZ. CVPR 16. [[Paper](https://semanticscholar.org/paper/d4310064dc18fbe72b0e424fdf474d3f13e2d650)]
  - 3D BBox, vehicle orientation, and shape as extra data

- Weakly Supervised Fine-Grained Categorization With Part-Based Image Representation. Zhang Y / Do M. A*SATR, SN. TIP 16. [[Paper](https://semanticscholar.org/paper/7a22cca5a339d11a5360850c7930c6449448525d)]
  - Convolutional filters for part proposal + Fisher Vector clusters for selecting useful parts + normalized FV concatenation from different scale parts

- Mining Discriminative Triplets of Patches for Fine-Grained Classification. Wang YM / Davis LS. U of Maryland, US. CVPR 16. [[Paper](https://semanticscholar.org/paper/41170023ef2b4bc9e0ab099708fdde92f4abb493)]
  - Triplets of patches with geometric constraints to aid mid-level representations

- Fully Convolutional Attention Networks for Fine-Grained Recognition. Liu X / Lin YQ. Baidu, CN. arXiv 16/03. [[Paper](https://semanticscholar.org/paper/9b678aa28facf4f90081d41c2c484c6addddb86d)]
  - Simultaneously compute parts without recursion using reinforcement learning


### 2015

- Bilinear CNN Models for Fine-grained Visual Recognition. Lin TY / Maji S. U of Massachusetts, US. ICCV 15. [[Paper](https://semanticscholar.org/paper/3d3f789a56dca288b2c8e23ef047a2b342184950)]
  - Outer product of (two) CNN feature maps (bilinear vector) as input to classifier

- Fine-Grained Recognition Without Part Annotations. Krause J / Fei-Fei L. Stanford U, US. CVPR 15. [[Paper](https://semanticscholar.org/paper/24129548ff77dad752a519f20bc8d14e03bb8397)]
  - Alignment by segmentation and pose graphs based on neighbors (highest cosine similarity of CNN features) to generate parts

- Part-Stacked CNN for Fine-Grained Visual Categorization. Huang SL / Zhang Y. U of Technology Sydney, AU. arXiv 15/12 / CVPR 16. [[Paper](https://semanticscholar.org/paper/e97d76194701469c6b138585292a01d4779009f5)]
  - Directly perform part-based classification on detected part locations from output feature maps using FCN, shared features, two stage training

- Deep LAC: Deep localization, alignment and classification for fine-grained recognition. Lin D / Jia JY. CVPR 15. Chinese U of Hong Kong, HK. [[Paper](https://semanticscholar.org/paper/154f7fbbf0eb5473c33daf062e37d138b16d9b1e)]
  - Backprop-able localization + alignment  based on templates (clustered from train set)

- Fine-Grained Categorization and Dataset Bootstrapping Using Deep Metric Learning with Humans in the Loop. Cui Y / Belongie S. Cornell U, US. arXiv 15/12 / CVPR 16. [[Paper](https://semanticscholar.org/paper/d62494053ec3bdccdc953d6916d2fab49b92049b)]
  - Triplet loss with sampling strategy for hard negatives and utilizing web data (CNN recognition-based + human verified)

- Multiple Granularity Descriptors for Fine-Grained Categorization. Wang DQ / Zhang Z. Fudan U, CN. ICCV 15. [[Paper](https://semanticscholar.org/paper/dd6b6beba7202deb1ceeb241438fdfd48e88b394)]
  - Detectors and classifiers for each level of class granularity / hierarchy

- Hyper-class Augmented and Regularized Deep Learning for Fine-grained Image Classification. Xie SN / Lin YQ. UC San Diego, US. CVPR 15. [[Paper](https://semanticscholar.org/paper/38128302a9d3b910e4f4fb7c3e17dc98f1d735b1)]
  - Web data and other datasets based on hyperclasses (dogs & orientation of cars) + auxiliary loss to predict hyperclasses

- Fine-Grained Image Classification by Exploring Bipartite-Graph Labels. Zhou F / Lin YQ. NEC Labs, US. arXiv 15/12 / CVPR 16. [[Paper](https://semanticscholar.org/paper/12bd9e03a1414deb09bf5d8d5c6ab98dd6a3347e)]
  - Jointly model fine-grained clases with pre-defined coarse classes (attributes / tags such as ingredients or macro-categories)

- A Fine-Grained Image Categorization System by Cellet-Encoded Spatial Pyramid Modeling. Zhang LM / Li XL. National U of Singapore, SN. TIE 15. [[Paper](https://semanticscholar.org/paper/43df4adb99cf4bee3daa8fbe67d864ac97d1f01d)]
  - Traditional encoding


### 2014

- DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition. Donahue J / Darrell T. UC Berkeley, US. ICML 2014. [[Paper](https://www.semanticscholar.org/paper/b8de958fead0d8a9619b55c7299df3257c624a96)]
  - CNN Features + Part Localization

- Part-based R-CNNs for Fine-grained Category Detection. Zhang N / Darrell T. UC Berkeley, US. CVPR 14. [[Paper](https://www.semanticscholar.org/paper/98bb60748eb8ef7a671cdd22faa87e377fd13060)]
  - Extends R-CNN to Detect Parts with Geometric Constraints

- Evaluation of Output Embeddings for Fine-Grained Image Classification. Akata Z / Schiele B. Max Planck Institute for Informatics, DE. arXiv 14/09 / CVPR 15. [[Paper](https://semanticscholar.org/paper/caccc069e658ea397c9faf673e74c959c734ff53)]
  - Learning from Web Text + Text-Based Zero-Shot Classification

- The application of two-level attention models in deep convolutional neural network for fine-grained image classification. Xiao TJ / Zhang Z. Peking U, CN. arXiv 14/11 / CVPR 15. [[Paper](https://semanticscholar.org/paper/7d5fd29a02f7fc50f7c36f1a74563d529620aa66)]
  - Bounding-Box Free Cropping (Weakly Supervised) via Multi-Stage Architecture

- Bird Species Categorization Using Pose Normalized Deep Convolutional Nets. Branson S / Perona P. Caltech, US. BMVC 14. [[Paper](https://www.semanticscholar.org/paper/23bdd2d82068419bf4923e6a0198fc0fa4468807)]
  - Pose-Normalized CNN + Fine-tuning

- Attention for Fine-Grained Categorization. Sermanet P / Real E. Google. arxiv 14/12 / ICLR 15 Workshop. [[Paper](https://semanticscholar.org/paper/6ef259c2f6d50373abfec14fcb8fa924f7b7af0b)]
  - Large-Scale Pretraining on CNN + RNN Attention for Weakly Supervised Crops

- Learning Category-Specific Dictionary and Shared Dictionary for Fine-Grained Image Categorization. Gao SH / Ma Y. Advanced Digital Sciences, SN. TIP 14. [[Paper](https://semanticscholar.org/paper/ce7dc05423589a7fe503abaddfa8654e695358fb)]
  - Category Specific and Shared Codebooks

- Jointly Optimizing 3D Model Fitting and Fine-Grained Classification. Lin YL / Davis LS. National Taiwan U, TW. ECCV 14. [[Paper](https://semanticscholar.org/paper/c1980e5d5c998ddec31cda9da148c354406a5eca)]
  - 3D Model Fitting as Auxiliary Task

- Fine-grained visual categorization via multi-stage metric learning. Qian Q / Lin YQ. Michigan State U, US. arXiv 14/02 / CVPR 15. [[Paper](https://semanticscholar.org/paper/e1c17cafe3b4bd5ddc78b523ec9eb0797b8f602f)]
  - Multi-Stage Distance Metric (Pull Positive Pairs and Push Negative Pairs, Contrastive-like) + KNN Classifier

- Revisiting the Fisher vector for fine-grained classification. Gosselin PH / Jegou H / Perronnin F. ETIS ENSEA / Inria, FR. Pattern Recognition Letters 2014. [[Paper](https://semanticscholar.org/paper/4589a778eab11277ac7e9cd3e78e68c56aa34ea3)]
  - Fisher Vector Scaling for FGIR

- Learning Features and Parts for Fine-Grained Recognition. Krause J / Fei-Fei L. Stanford U, US. CVPR 14. [[Paper](https://semanticscholar.org/paper/a10f734e30d8dcb8506c9ea5b1074e6c668904e2)]
  - CNN + Unsupervised Part Discovery for Focusing on CNN Regions (No multi-stage)

- Nonparametric Part Transfer for Fine-Grained Recognition. Goring C / Denzler J. University Jena, DE. CVPR 14. [[Paper](https://semanticscholar.org/paper/fbc06216b01e415e60c67a6028bd1487faa42f19)]
  - Train images with similar shape to current image then transfer part annotations


### 2013

- POOF: Part-Based One-vs.-One Features for Fine-Grained Categorization, Face Verification, and Attribute Estimation. Berg T / Belhumeur P. Columbia U, US. CVPR 13. [[Paper](https://semanticscholar.org/paper/23efd4b0aef1ae0b356fe88141da085526ed3df0)]
  - Align two images, divide into small patches, classify and distinguish between patches, select most discriminative then classify again

- Fine-Grained Crowdsourcing for Fine-Grained Recognition. Deng J / Fei-Fei L. CVPR 13. [[Paper](https://semanticscholar.org/paper/526eff11d1f545d0dafe025f9c0d5d558456f624)]
  - Crowdsource discriminative regions and algorithm to make use of them

- Symbiotic Segmentation and Part Localization for Fine-Grained Categorization. Chai YN / Zisserman A. U of Oxford, UK. ICCV 13. [[Paper](https://semanticscholar.org/paper/0a26477bd1e302219a065eea16495566ead2cede)]
  - Joint loss for parts + foreground / background segmentation

- Deformable Part Descriptors for Fine-Grained Recognition and Attribute Prediction. Zhang N / Darrel T. UC Berkeley, US. ICCV 13. [[Paper](https://semanticscholar.org/paper/19c9ac899d5c1a008eaee887556bc1b61ff8132e)]
  - Part localization + pose normalization

- Efficient Object Detection and Segmentation for Fine-Grained Recognition. Angelova A / Zhu SH. NEC Labs America, US. CVPR 13. [[Paper](https://semanticscholar.org/paper/21256be13869da1c98160e3498209daa6497d99c)]
  - Detect and segment object then crop

- Fine-Grained Categorization by Alignments. Gavves E / Tuytelaars T. U of Amsterdam, ICCV 13. [[Paper](https://semanticscholar.org/paper/02d94a8dfa64680bf07fc96aec5548b2793001aa)]
  - Align images then predict parts based on similar images in train set

- Style Finder : Fine-Grained Clothing Style Recognition and Retrieval. Di W / Sundaresan N. UC San Diego, US. CVPR Workshop 13. [[Paper](https://semanticscholar.org/paper/448efcae3b97aa7c01b15c6bc913d4fbb275f644)]
  - Clothing dataset

- Hierarchical Part Matching for Fine-Grained Visual Categorization. Xie LX / Zhang B. Tsinghua U, CN. ICCV 13. [[Paper](https://semanticscholar.org/paper/4a41ad3299625f8b30b11593d02c1c2fdebe7aa7)]
  - Segmentation into semantic parts + combining mid-level features

- Multi-level Discriminative Dictionary Learning towards Hierarchical Visual Categorization. Shen L / Huang QM. U of Chinese Academy of Sciences, CN. CVPR 13. [[Paper](https://semanticscholar.org/paper/7a0095a853f28f750337ee495df2286e337847df)]
  - Hierarchical classification

- Vantage Feature Frames for Fine-Grained Categorization. Sfar A / Geman D. INRIA Saclay. CVPR 13. [[Paper](https://semanticscholar.org/paper/2ba5e4c421b1413139e4bc5d935d6d48cc753757)]
  - Find points and orientation from which to distinguish fine-grained details (inspired by experts approach)

- Con-text: text detection using background connectivity for fine-grained object classification. Karaoglu S / Gevers T. U of Amsterdam, NL. ACM MM 13. [[Paper](https://semanticscholar.org/paper/fc36ab26877604d782c3983a438079aef7686ebf)]
  - Text detection (foreground) by reconstructing background using morphology then substract background


### 2012

- Discovering localized attributes for fine-grained recognition. Duan K / Grauman K. Indiana U, US. CVPR 12. [[Paper](https://semanticscholar.org/paper/0182d090478be67241392df90212d6cd0fb659e6)]
  - Detection of human interpretable attributes

- Unsupervised Template Learning for Fine-Grained Object Recognition. Shapiro L / Yang SL. U of Washington, US. NIPS 12. [[Paper](https://semanticscholar.org/paper/be944ae102ad3b09d34ad9217ee2f6829097e547)]
  - Template detection and use them to align images

- A codebook-free and annotation-free approach for fine-grained image categorization. Yao BP / Fei-Fei L. Stanford U, US. CVPR 12. [[Paper](https://semanticscholar.org/paper/a797ed714ed3d65f9174a0f2b33f605748283681)]
  - Template-based similarity matching between random templates


### 2011

- Combining randomization and discrimination for fine-grained image categorization. Yao BP. / Fei-Fei L. Stanford U, US. CVPR 11. [[Paper](https://semanticscholar.org/paper/4e7a689b57415342cd6c6dc57b4d0074868e8042)]
  - Random forest + discriminative trees

- Fisher Vectors for Fine-Grained Visual Categorization. Sanchez J / Akata Z. Xerox. FGVC Workshop in CVPR 11. [[Paper](https://semanticscholar.org/paper/e771d47d39c2d94dfa18a95ffe8d5d631b7ff9d2)]
  - Fisher vectors

## Datasets

- SkinCon: A skin disease dataset densely annotated by domain experts for fine-grained model debugging and analysis [[Paper](https://semanticscholar.org/paper/6044a72c240f6eda2fbf55d19fe3e945db648b42)]

- GrainSpace: A Large-scale Dataset for Fine-grained and Domain-adaptive Recognition of Cereal Grains [[Paper](https://semanticscholar.org/paper/09fded76f4a64e13de81e247548f47af0032ba8f)]

- FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery [[Paper](https://semanticscholar.org/paper/6d3c2dc63ff0deec10f60e5a515c93af4f8676f2)]

- Yoga-82: A New Dataset for Fine-grained Classification of Human Poses [[Paper](https://semanticscholar.org/paper/910bbdf7b1b7f22b44b9cc777ad088486517d194)]

- Building a bird recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection [[Paper](https://semanticscholar.org/paper/6dd6d184005388f75c411f27e224b2b6b87b7c60)]

- A large-scale car dataset for fine-grained categorization and verification [[Paper](https://semanticscholar.org/paper/496c6db97a4f16e6bc9e4d63af420eb66f900f74)]

- Birdsnap: Large-Scale Fine-Grained Visual Categorization of Birds [[Paper](https://semanticscholar.org/paper/cc46fc50cbae566b89fa0cf2f8fc7bd81d901f31)]

- 3D Object Representations for Fine-Grained Categorization [[Paper](https://semanticscholar.org/paper/a83cec6a91701bd8500f8c43ad731d4353c71d55)]

- Fine-Grained Visual Classification of Aircraft [[Paper](https://semanticscholar.org/paper/522d65a3db7431015aeaa201a7fc4450a57e40c3)]

- Novel Dataset for Fine-Grained Image Categorization : Stanford Dogs [[Paper](https://semanticscholar.org/paper/b5e3beb791cc17cdaf131d5cca6ceb796226d832)]


## FGIR-OSI


### Acknowledgement

Thanks [Awesome-Crowd-Counting](https://github.com/gjy3035/Awesome-Crowd-Counting) for the template.
