# CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment

**会议**: 投稿中  
**arXiv**: [2603.12722](https://arxiv.org/abs/2603.12722)  
**代码**: 未提及  
**领域**: 3D视觉  
**关键词**: cognitioncapturerpro, towards, high-fidelity, visual, decoding  

## 一句话总结
我们提出了 CognitionCapturerPro，这是一种通过协作训练将脑电图与多模态先验（图像、文本、深度和边缘）集成的增强框架。
## 核心问题
由于保真度损失和表征偏移，脑电图的视觉刺激重建仍然具有挑战性。
## 关键方法
1. 我们提出了 CognitionCapturerPro，这是一个增强的框架，通过协作训练将脑电图与多模态先验（图像、文本、深度和边缘）集成起来
2. 我们的核心贡献包括用于量化特定模态保真度的不确定性加权相似性评分机制和用于集成共享表示的融合编码器
3. 通过采用简化的对齐模块和预训练的扩散模型，我们的方法在 THINGS-EEG 数据集上显着优于原始 CognitionCapturer，将 Top-1 和 Top-5 检索精度分别提高了 25.9% 和 10.6%
4. 代码位于：https://github.com/XiaoZhangYES/CognitionCapturerPro。
## 亮点 / 我学到了什么
- 通过采用简化的对齐模块和预训练的扩散模型，我们的方法在 THINGS-EEG 数据集上显着优于原始 CognitionCapturer，将 Top-1 和 Top-5 检索精度分别提高了 25.9% 和 10.6%
## 局限性 / 可改进方向
- 需要详细阅读原文以确认具体局限
- 潜在扩展方向待进一步分析

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见原文 | 详见原文 | — | — | — |

## 与我的研究方向的关联
- `causal_3d_occupancy`
- `open_vocab_3d_occupancy`
- `privacy_3d_scene`

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
