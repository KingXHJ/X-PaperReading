# 计算机视觉-3D


## 目录
- [笔记](#笔记)
- [立体视觉网络排行网站](https://vision.middlebury.edu/stereo/)
- [双目视觉数据网站](./papers/双目立体开源数据集资源汇总.pdf)
- [MVTec anomaly detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
- [Carlos Hernandez research result](https://www.carlos-hernandez.org/publications.html)
- [返回上一层 README](../README.md)


## 笔记

| 笔记 | 年份 | 名字                                                         | 简介                 | 引用 |
| ------ | ---- | ------------------------------------------------------------ | -------------------- | ------------------------------------------------------------ |
| [✅](./papers/MVSNet%20Depth%20Inference%20for%20Unstructured%20Multi-view%20Stereo.md)      | 2018 | [MVSNet](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.pdf) | 第一个基于深度学习的MVS模型                   | [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F87ca28235555f7e70cf1edc2a63cda4aef7fee42%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/MVSNet%3A-Depth-Inference-for-Unstructured-Multi-view-Yao-Luo/87ca28235555f7e70cf1edc2a63cda4aef7fee42) |
| [✅](./papers/Cost%20Volume%20Pyramid%20Based%20Depth%20Inference%20for%20Multi-View%20Stereo.md)      | 2020 | [CVP-MVSNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Cost_Volume_Pyramid_Based_Depth_Inference_for_Multi-View_Stereo_CVPR_2020_paper.pdf) | 从粗糙到精细的代价金字塔，精度高，速度慢                   | [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F73fdd0c9c0a4f6e07fda16449db1fe703c13ef23%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Cost-Volume-Pyramid-Based-Depth-Inference-for-Yang-Mao/73fdd0c9c0a4f6e07fda16449db1fe703c13ef23) |
| [✅](./papers/PatchmatchNet%20Learned%20Multi-View%20Patchmatch%20Stereo.md)      | 2020 | [PatchmatchNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_PatchmatchNet_Learned_Multi-View_Patchmatch_Stereo_CVPR_2021_paper.pdf) | 借鉴传统PatchMatch算法，采用金字塔思想，精度高，速度快                   | [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb9ec0bb70a2425493f187ccaf8ea0461e90a7381%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/PatchmatchNet%3A-Learned-Multi-View-Patchmatch-Stereo-Wang-Galliani/b9ec0bb70a2425493f187ccaf8ea0461e90a7381) |
| [✅](./papers/Self-supervised%20Multi-view%20Stereo%20via%20Effective%20Co-Segmentation%20and%20Data-Augmentation.md)      | 2021 | [JDACS-MS](https://arxiv.org/pdf/2104.05374.pdf) | 无监督网络解决不同视角下颜色不一致问题                   | [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9da0ab9744700e31eef504403ad872cb99ec4fd0%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Self-supervised-Multi-view-Stereo-via-Effective-and-Xu-Zhou/9da0ab9744700e31eef504403ad872cb99ec4fd0) |
| [✅](./papers/Object%20as%20Query%20Equipping%20Any%202D%20Object%20Detector%20with%203D%20Detection%20Ability.md)      | 2023 | [MV2D](https://arxiv.org/pdf/2301.02364.pdf) | 多视图2D物体引导的3D物体检测器题                   | [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F3c43e940290452e6b56bd1a736a6d745c0f30c90%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Object-as-Query%3A-Equipping-Any-2D-Object-Detector-Wang-Huang/3c43e940290452e6b56bd1a736a6d745c0f30c90) |
| [✅](./papers/Model-Agnostic%20Hierarchical%20Attention%20for%203D%20Object%20Detection.md)      | 2023 | [Model-Agnostic Hierarchical Attention for 3D Object Detection](https://arxiv.org/pdf/2301.02650.pdf) | 点云网络利用Transformers + "多尺度"/"尺寸自适应局部"注意力机制实现更好的小目标的3D目标检测题                   | [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F620da8851e38e932e62fb2ee4a28a13cfeb5772f%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Model-Agnostic-Hierarchical-Attention-for-3D-Object-Shu-Xue/620da8851e38e932e62fb2ee4a28a13cfeb5772f) |
| [✅](./papers/Super%20Sparse%203D%20Object%20Detection.md)      | 2023 | [Super Sparse 3D Object Detection](https://arxiv.org/pdf/2301.02562.pdf) | 点云网络利用超稀疏特征实现远距离3D目标检测题                   | [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F55d2665d77965dad3e6cd699d523dd326195e385%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Super-Sparse-3D-Object-Detection-Fan-Yang/55d2665d77965dad3e6cd699d523dd326195e385) |


*[跳转至目录](#目录)*