# Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation

**会议**: arXiv 2026  
**arXiv**: [2603.12581](https://arxiv.org/abs/2603.12581)  
**作者**: Jianqiang Lin, Zhiqiang Shen, Peng Cao, Jinzhu Yang, Osmar R. Zaiane
**代码**: [https://github.com/ziyi-start/MSG-LDM](https://github.com/ziyi-start/MSG-LDM)  
**领域**: 多模态/VLM  
**关键词**: multiscale, structure-guided, latent, diffusion, multimodal  

## 一句话总结
尽管扩散模型在多模态磁共振成像（MRI）翻译任务中取得了显着进展，但现有方法在处理任意模态缺失场景时仍然容易出现解剖学不一致或纹理细节退化的问题。
## 背景与动机
Although diffusion models have achieved remarkable progress in multi-modal magnetic resonance imaging (MRI) translation tasks, existing methods still tend to suffer from anatomical inconsistencies or degraded texture details when handling arbitrary missing-modality scenarios.. To address these issues, we propose a latent diffusion-based multi-modal MRI translation framework, termed MSG-LDM.

## 核心问题
为了解决这些问题，我们提出了一种基于潜在扩散的多模态 MRI 翻译框架，称为 MSG-LDM。
## 方法详解

### 整体框架
- To address these issues, we propose a latent diffusion-based multi-modal MRI translation framework, termed MSG-LDM.
- By leveraging the available modalities, the proposed method infers complete structural information, which preserves reliable boundary details.
- Specifically, we introduce a style--structure disentanglement mechanism in the latent space, which explicitly separates modality-specific style features from shared structural representations, and jointly models low-frequency anatomical layouts and high-frequency boundary details in a multi-scale feature space.
- During the structure disentanglement stage, high-frequency structural information is explicitly incorporated to enhance feature representations, guiding the model to focus on fine-grained structural cues while learning modality-invariant low-frequency anatomical representations.

### 关键设计
1. **关键组件1**: Specifically, we introduce a style--structure disentanglement mechanism in the latent space, which explicitly separates modality-specific style features from shared structural representations, and jointly models low-frequency anatomical layouts and high-frequency boundary details in a multi-scale feature space.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 尽管扩散模型在多模态磁共振成像（MRI）翻译任务中取得了显着进展，但现有方法在处理任意模态缺失场景时仍然容易出现解剖学不一致或纹理细节退化的问题。
- 此外，为了减少特定模态样式的干扰并提高结构表示的稳定性，我们设计了样式一致性损失和结构感知损失。
- 在 BraTS2020 和 WMH 数据集上进行的大量实验表明，所提出的方法优于现有的 MRI 合成方法，特别是在重建完整结构方面。
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
