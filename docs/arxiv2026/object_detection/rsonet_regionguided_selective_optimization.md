# RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection

**会议**: arXiv 2026  
**arXiv**: [2603.12685](https://arxiv.org/abs/2603.12685)  
**作者**: Bin Wan, Runmin Cong, Xiaofei Zhou, Hao Fang, Chengtao Lv
**代码**: 待确认  
**领域**: 目标检测  
**关键词**: rsonet, region-guided, selective, optimization, network  

## 一句话总结
本文重点研究 RGB 图像和热图像之间显着区域的不一致。

## 背景与动机
This paper focuses on the inconsistency in salient regions between RGB and thermal images.. To address this issue, we propose the Region-guided Selective Optimization Network for RGB-T Salient Object Detection, which consists of the region guidance stage and saliency generation stage.

## 核心问题
为了解决这个问题，我们提出了用于 RGB-T 显着目标检测的区域引导选择性优化网络，该网络由区域引导阶段和显着性生成阶段组成。

## 方法详解

### 整体框架
- To address this issue, we propose the Region-guided Selective Optimization Network for RGB-T Salient Object Detection, which consists of the region guidance stage and saliency generation stage.
- In the region guidance stage, three parallel branches with same encoder-decoder structure equipped with the context interaction (CI) module and spatial-aware fusion (SF) module are designed to generate the guidance maps which are leveraged to calculate similarity scores.
- We conduct extensive experiments on the RGB-T dataset, and the results demonstrate that the proposed RSONet achieves competitive performance against 27 state-of-the-art SOD methods.

### 关键设计
1. **关键组件1**: In the region guidance stage, three parallel branches with same encoder-decoder structure equipped with the context interaction (CI) module and spatial-aware fusion (SF) module are designed to generate the guidance maps which are leveraged to calculate similarity scores.
2. **关键组件2**: Then, in the saliency generation stage, the selective optimization (SO) module fuses RGB and thermal features based on the previously obtained similarity values to mitigate the impact of inconsistent distribution of salient targets between the two modalities.
3. **关键组件3**: After that, to generate high-quality detection result, the dense detail enhancement (DDE) module which adopts the multiple dense connections and visual state space blocks is applied to low-level features for optimizing the detail information.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 我们对 RGB-T 数据集进行了广泛的实验，结果表明，所提出的 RSONet 与 27 种最先进的 SOD 方法相比，实现了具有竞争力的性能。

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
