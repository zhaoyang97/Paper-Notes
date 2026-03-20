# Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions

**会议**: 投稿中  
**arXiv**: [2603.12468](https://arxiv.org/abs/2603.12468)  
**代码**: 未提及  
**领域**: 医学图像  
**关键词**: adaptation, weakly, supervised, localization, histopathology  

## 一句话总结
本文介绍了 \sfdadep，这是一种受机器遗忘启发的方法，它将 SFDA 表述为识别和纠正预测偏差的迭代过程。
## 核心问题
然而，由于它们依赖于自我训练，因此初始偏差在训练迭代中得到加强，从而降低了分类和定位任务的性能。
## 关键方法
1. 本文介绍了 \sfdadep，这是一种受机器遗忘启发的方法，它将 SFDA 表述为识别和纠正预测偏差的迭代过程
2. 当部署在目标域中时，分布变化仍然是性能下降的主要原因，特别是当应用于具有不同染色协议和扫描仪特性的新器官或机构时
3. 在更强的跨域转移下，WSOL 预测可能会偏向于主导类别，从而在目标域中产生高度倾斜的伪标签分布
4. 无源（无监督）域适应（SFDA）方法通常用于解决域转移问题
## 亮点 / 我学到了什么
- 使用多个 WSOL 模型对跨器官和中心组织病理学基准（glas、CAMELYON-16、CAMELYON-17）进行的广泛实验表明，SFDA-DeP 持续改进了最先进的 SFDA 基线的分类和定位
## 局限性 / 可改进方向
- 需要详细阅读原文以确认具体局限
- 潜在扩展方向待进一步分析

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见原文 | 详见原文 | — | — | — |

## 与我的研究方向的关联
- `2d_to_3d_medical_distill`
- `medical_bias_audit`
- `medical_dynamic_routing`

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
