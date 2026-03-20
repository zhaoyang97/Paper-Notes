# LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models

**会议**: ICCV 2025  
**arXiv**: [2403.15388](https://arxiv.org/abs/2403.15388)  
**代码**: [https://llava-prumerge.github.io/](https://llava-prumerge.github.io/)  
**领域**: 多模态VLM / 模型加速  
**关键词**: token reduction, visual token pruning, token merging, LMM效率, CLS注意力稀疏性  

## 一句话总结
利用CLIP-ViT中[CLS] token与视觉token之间注意力分数的稀疏特性，通过IQR异常值检测自适应选择重要视觉token，再用k-近邻聚类将被剪除token的信息合并回保留token，实现视觉token 14倍压缩且性能几乎不降。

## 背景与动机
LMM（如LLaVA-1.5）将数百个视觉token（如576个）作为前缀输入LLM，而LLM的注意力计算复杂度与输入长度的平方成正比。随着高分辨率图像和视频的支持，视觉token数量进一步膨胀（Video-LLaVA用2048个token）。现有的加速方法主要是压缩LLM本身（用更小的LLM或量化），但忽略了一个关键事实——大量视觉token是冗余的。

## 核心问题
能否在不损失LMM推理能力的前提下，大幅减少视觉encoder输出到LLM的视觉token数量？核心挑战在于如何判断哪些token重要、如何保留被剪除token中的有用信息。

## 方法详解

### 整体框架
PruMerge是一个即插即用的模块，位于视觉encoder输出之后、MLP投影层之前。分三步：(1) 用CLS注意力稀疏性通过IQR异常值检测自适应选择重要token；(2) 用k-近邻将被剪除token聚类到最近的保留token；(3) 按注意力权重加权平均合并每个聚类。PruMerge+在此基础上额外补充均匀采样的空间token。

### 关键设计
1. **基于CLS注意力稀疏性的自适应选择（AITS）**：观察到CLIP-ViT倒数第二层中[CLS] token与视觉token的注意力分数呈高度稀疏分布——绝大多数token的注意力接近零，只有少数token有显著高的注意力值。利用统计学中的IQR（四分位距）方法检测"异常值"：计算Q1和Q3，超过Q3 + 1.5×IQR的token被视为重要token保留。这使得token选择数量是自适应的——复杂图像（如含大量文字）保留更多token，简单图像保留更少。

2. **基于Key相似度的token合并（TS）**：被剪除的token虽然不够"重要"，但可能包含补充信息（如大面积背景区域）。用ViT最后一层的Key向量计算token间相似度（$\text{Sim}(y_i, y_j) = k_i \cdot k_j^T$），找每个保留token的k个最近邻（被剪token），然后按CLS注意力值加权平均合并到保留token中。这比图匹配方法（如CrossGet的$O(n^2)$）更高效（$O(n)$）。

3. **PruMerge+的空间采样补充**：纯注意力选择可能遗漏某些区域的信息。PruMerge+在IQR选择的token基础上，额外均匀采样空间位置的token，确保全图覆盖。压缩比从14x降到4x，但性能更接近原始模型。

### 损失函数 / 训练策略
PruMerge本身不需要训练——完全training-free可直接应用。但作者发现用LoRA微调1个epoch可以进一步提升性能（让LLM适应新的token结构）。在Video-LLaVA上甚至完全training-free就能提升性能。

## 实验关键数据
| 方法 | LLM | Token数 | VQAv2 | SQAI | TextVQA | POPE | MME | MMB |
|------|-----|---------|-------|------|---------|------|-----|-----|
| LLaVA-1.5 | 7B | 576 | 78.5 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 |
| +PruMerge | 7B | ~32 | 72.0 | **68.5** | 56.0 | 76.3 | 1350.3 | 60.9 |
| +PruMerge+ | 7B | ~144 | 76.8 | **68.3** | 57.1 | 84.0 | 1462.4 | **64.9** |

- PruMerge用平均32个token（5.5%）即可维持大部分性能，ScienceQA上甚至超过原始模型
- PruMerge+用25%的token基本追平原始模型，MMB上略优
- FLOPs减少约10x（9.3TB→0.91TB），prefill时间从88.6ms降到15.3ms
- 在Video-LLaVA上无需训练直接提升ActivityNet-QA 3分
- 对比ToMe/ATS/EViT等单模态方法，PruMerge+大幅领先（POPE上84.0 vs 51.0/57.4/60.1）

### 消融实验要点
- AITS选择模块贡献最大（是核心设计），TS合并模块进一步提升约2分
- PruMerge自适应性强：TextVQA/MME平均40 token，POPE平均35 token，SQA仅16 token
- IQR方法优于所有对比的token采样策略（顺序/均匀空间采样）
- LoRA微调比training-free进一步提升2-3分

## 亮点
- **自适应机制非常优雅**：利用IQR异常值检测让token数量随图像复杂度自动调整，统计学方法在深度学习中的巧妙应用
- **洞察深刻**：CLS注意力稀疏性→大多数视觉token与全局语义关联弱→可以安全剪掉
- **极其实用**：即插即用、training-free、与量化正交可叠加、适用于图像和视频LMM
- **Prune+Merge的结合**：先剪枝保留重要的，再合并回收被剪的——不浪费任何信息

## 局限性 / 可改进方向
- PruMerge（14x压缩）在需要细粒度理解的任务上有明显掉点（VQAv2下降6.5%，POPE下降9.6%）
- IQR方法假设CLS注意力稀疏分布，对非CLIP编码器的适用性未验证
- 没有考虑文本query与视觉token的关系——重要性判断完全基于视觉端，忽视了不同问题可能需要不同视觉区域
- 仅在LLaVA-1.5上验证，未测试更新的模型（如LLaVA-NeXT、Qwen-VL）

## 与相关工作的对比
- **vs. Feather the Throttle**：Feather在LLM内部剪枝（FastV策略），PruMerge在encoder出口剪枝；Feather发现了RoPE偏差问题，PruMerge利用CLS稀疏性。两者动机不同但互补
- **vs. FastV**：FastV在LLM浅层剪枝，PruMerge在进入LLM前剪枝，前置更彻底；FastV用注意力排序，PruMerge用IQR自适应
- **vs. ToMe/EViT/ATS**：这些是单模态ViT加速方法，在LMM场景下效果差因为最终输出的是[CLS] token而非全部token

## 启发与关联
- CLS注意力稀疏性的发现与Feather the Throttle中的RoPE偏差发现形成互补，值得深入研究视觉token的注意力模式
- 自适应token数量的思路可以扩展到更多场景——难样本多token、简单样本少token
- 与Scaling Language-Free Visual Repr的研究结合，考虑SSL encoder的token压缩特性

## 评分
- 新颖性: ⭐⭐⭐⭐ CLS稀疏性→IQR异常检测→自适应选择的思路链顺畅有洞察
- 实验充分度: ⭐⭐⭐⭐ 6个benchmark + 视频扩展 + 多方法对比 + 详尽消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，可视化（Figure 1/3）直观有说服力
- 价值: ⭐⭐⭐⭐ 高实用价值的即插即用方案，但ICCV2025时已有更新的竞争方法
