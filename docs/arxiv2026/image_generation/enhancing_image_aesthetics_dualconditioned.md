# Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception

**会议**: arXiv 2026  
**arXiv**: [2603.11556](https://arxiv.org/abs/2603.11556)  
**作者**: Xinyu Nan, Ning Wang, Yuyao Zhai, Mei Yang
**代码**: 待确认  
**领域**: 扩散模型/生成  
**关键词**: enhancing, image, aesthetics, dual-conditioned, diffusion  

## 一句话总结
图像美感增强旨在感知图像中的美感缺陷并进行相应的编辑操作，具有很高的挑战性，要求模型具备创造力和美感感知能力。
## 背景与动机
Image aesthetic enhancement aims to perceive aesthetic deficiencies in images and perform corresponding editing operations, which is highly challenging and requires the model to possess creativity and aesthetic perception capabilities.. Although recent advancements in image editing models have significantly enhanced their controllability and flexibility, they struggle with enhancing image aesthetic.

## 核心问题
尽管图像编辑模型的最新进展显着增强了其可控性和灵活性，但它们在增强图像美感方面遇到了困难。
## 方法详解

### 整体框架
- In this paper, we propose Dual-supervised Image Aesthetic Enhancement (DIAE), a diffusion-based generative model with multimodal aesthetic perception.
- To better leverage the weak matching characteristics of IIAEData during training, a dual-branch supervision framework is also introduced for weakly supervised image aesthetic enhancement.

### 关键设计
1. **关键组件1**: The primary challenges are twofold: first, following editing instructions with aesthetic perception is difficult, and second, there is a scarcity of "perfectly-paired" images that have consistent content but distinct aesthetic qualities.
2. **关键组件2**: First, DIAE incorporates Multimodal Aesthetic Perception (MAP) to convert the ambiguous aesthetic instruction into explicit guidance by (i) employing detailed, standardized aesthetic instructions across multiple aesthetic attributes, and (ii) utilizing multimodal control signals derived from text-image pairs that maintain consistency within the same aesthetic attribute.
3. **关键组件3**: Second, to mitigate the lack of "perfectly-paired" images, we collect "imperfectly-paired" dataset called IIAEData, consisting of images with varying aesthetic qualities while sharing identical semantics.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 实验结果表明，DIAE 优于基线，并获得了优异的图像美学分数和图像内容一致性分数。
## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
