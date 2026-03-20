# GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning

**会议**: CVPR 2026  
**arXiv**: [2603.13370](https://arxiv.org/abs/2603.13370)  
**代码**: [https://github.com/oamyjin/GraphVLM](https://github.com/oamyjin/GraphVLM) (有)  
**领域**: 多模态VLM / 图学习  
**关键词**: VLM图学习, 多模态图, 三范式评估, 结构化推理, 基准测试  

## 一句话总结
提出 GraphVLM，首个系统评估 VLM 在多模态图学习（MMGL）中能力的基准，研究三种互补范式——VLM-as-Encoder（特征融合增强 GNN）、VLM-as-Aligner（潜空间/语言空间桥接模态）、VLM-as-Predictor（VLM 直接作为图学习 backbone），在 6 个跨领域数据集上验证，发现 VLM-as-Predictor 范式效果最好，揭示了 VLM 作为多模态图学习新基础的巨大潜力。

## 背景与动机
VLM 在对齐和理解多模态信号上表现出色，但其对结构化数据的推理能力（多模态实体通过显式关系图连接）尚未被系统探索。现实应用如社交网络、推荐系统和科学发现中，多模态信息天然具有图结构。

## 核心问题
如何系统评估和利用 VLM 进行多模态图学习？三种 VLM 集成范式各有何优劣？

## 方法详解

### 三种范式

1. **VLM-as-Encoder**: 用 VLM 提取多模态节点特征，然后输入传统 GNN 进行图推理。VLM 提供更丰富的节点表示。

2. **VLM-as-Aligner**: 在潜空间或语言空间中桥接不同模态，为基于 LLM 的结构化推理提供对齐的表示。

3. **VLM-as-Predictor**: 直接将 VLM 作为多模态 backbone 进行图学习任务——效果最好且最一致。

## 实验关键数据
- 6 个跨领域数据集
- VLM-as-Predictor 在所有范式中表现最好且最稳定

## 亮点
- **系统性基准**：首次对 VLM 在图学习中的能力进行全面评估
- **三范式对比**：为选择 VLM 集成方式提供了实证指导
- **VLM-as-Predictor 的发现**：揭示了 VLM 直接做图学习的巨大潜力

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个 VLM+图学习的系统基准
- 实验充分度: ⭐⭐⭐⭐ 6 数据集 3 范式全面对比
- 写作质量: ⭐⭐⭐⭐ 摘要清晰
- 价值: ⭐⭐⭐⭐ 为 VLM 在结构化数据上的应用提供了基础框架
