# CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation

**会议**: 投稿中  
**arXiv**: [2603.12008](https://arxiv.org/abs/2603.12008)  
**代码**: 未提及  
**领域**: 语义分割  
**关键词**: crossearth-sar, sar-centric, billion-scale, geospatial, foundation  

## 一句话总结
为了解决这个问题，我们提出了 CrossEarth-SAR，这是第一个十亿级 SAR 视觉基础模型，建立在一种新颖的物理引导稀疏专家混合 (MoE) 架构之上，该架构结合了物理描述符，专门为跨域语义分割而设计。
## 核心问题
然而，由于成像机制不同，传感器和区域之间的域转移严重阻碍了其语义泛化。
## 关键方法
1. 为了解决这个问题，我们提出了 CrossEarth-SAR，这是第一个十亿级 SAR 视觉基础模型，建立在一种新颖的物理引导稀疏专家混合 (MoE) 架构之上，该架构结合了物理描述符，专门为跨域语义分割而设计
2. 然而，由于成像机制不同，传感器和区域之间的域转移严重阻碍了其语义泛化
3. 为了促进大规模预训练，我们开发了 CrossEarth-SAR-200K，这是一个弱且完全监督的数据集，统一了公共和私人 SAR 图像
4. 我们还推出了一个基准套件，其中包含 8 个不同领域间隙的 22 个子基准，为 SAR 图像领域泛化语义分割建立了第一个统一标准
## 亮点 / 我学到了什么
- 大量实验表明，CrossEarth-SAR 在 20 个基准上取得了最先进的结果，在多间隙传输下的一些基准上超越了之前的方法超过 10\% mIoU
## 局限性 / 可改进方向
- 需要详细阅读原文以确认具体局限
- 潜在扩展方向待进一步分析

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见原文 | 详见原文 | — | — | — |

## 与我的研究方向的关联
- `progressive_seg_distill`
- `unified_freq_prompt_sam`
- `report_to_segmentation`

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
