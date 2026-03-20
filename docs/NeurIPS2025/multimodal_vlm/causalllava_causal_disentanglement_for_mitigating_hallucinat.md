# Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.19474](https://arxiv.org/abs/2505.19474)  
**代码**: [https://github.com/IgniSavium/Causal-LLaVA](https://github.com/IgniSavium/Causal-LLaVA)  
**领域**: 多模态VLM  
**关键词**: hallucination, causal inference, disentanglement, MLLM, co-occurrence bias, backdoor adjustment  

## 一句话总结
揭示 MLLM 中物体幻觉的表示层根因——数据集共现偏差导致的语义纠缠，提出双路因果解纠缠框架（Causal-Driven Projector + Causal Intervention Module），通过后门调整在 projector 和最终 Transformer 层分离共现物体表示，使 MME-Perception 提升 22.6%。

## 背景与动机
MLLM 的物体幻觉（描述不存在的物体）主要源于训练数据中的共现偏差——如"餐桌"总与"椅子"同框出现，模型学会了这种虚假关联。但先前工作仅在**统计层面**验证了共现频率与幻觉率的相关性，未深入探究表示空间中的机制。

本文的关键发现：通过可视化 LLaVA 各层的物体表示（PCA），发现：
- CLIP 编码器输出时物体表示分散（正常）
- 经 Projector 处理后高频共现物体表示急剧聚拢（纠缠形成）
- 纠缠持续传播至 LLM 的 1-15 层（理解阶段）
- 在最终第 40 层（预测阶段）仍保留显著纠缠

这意味着 projector 是偏差注入的关键节点，且偏差一旦形成就贯穿整个推理过程。

## 核心问题
如何从**表示学习层面**阻断共现偏差的传播，使共现物体的语义表示解纠缠，从而减少物体幻觉？

## 方法详解

### 整体框架
在 LLaVA 的两个关键位置插入因果干预模块：(1) Projector 后（阻断视觉混淆因子 $D_v$ 向 soft tokens $S$ 的传播），(2) LLM 最终 Transformer 层（阻断视觉/文本混淆因子 $D_v$/$D_t$ 向预测 $W$ 的传播）。

### 关键设计
1. **Causal-Driven Projector**: 基于后门调整公式 $P(Y|do(X)) \approx_{NWGM} g_f(f_v) + g_z(\mathbb{E}_z[z])$，将原始 projector 输出与混淆因子期望值的估计相加。混淆因子字典 $D \in \mathbb{R}^{K \times \sigma}$ 由 80 类 COCO 物体的平均 post-projector 视觉表示构成（从 5000 样本聚合），通过交叉注意力动态估计 $\mathbb{E}_z[z]$。

2. **Causal Intervention Module（LLM 层）**: 在最终 Transformer 层插入，分别用视觉和文本混淆因子字典 $D_v, D_t$ 做交叉注意力干预：$\text{CausalIntervention}(h) = \text{CrossAttn}(h, D_v, D_v) + \text{CrossAttn}(h, D_t, D_t)$，解耦隐状态中的视觉和文本共现偏差。

3. **NWGM 近似**: 因果干预的精确计算需遍历所有混淆因子，计算量大。利用 Normalized Weighted Geometric Mean 将 Softmax 外的期望移入内部，简化为 $\text{Softmax}[g(x, \mathbb{E}_z[z])]$。

### 损失函数 / 训练策略
保持 LLaVA 原始训练配置，仅修改：batch 256（2x）、lr 1e-3（0.5x）。混淆因子字典从非因果模型训练 0.1 epoch 的 checkpoint 中提取。8×H20 GPU 训练，混淆因子估计额外 ~1 小时。

## 实验关键数据

| 模型 | LLM | POPE_rnd | MME_P | CHAIR_s↓ | CHAIR_i↓ |
|------|-----|----------|-------|----------|----------|
| LLaVA | LLaMA-2-7B | 71.70 | 714.29 | 33.0 | 9.5 |
| **Causal-LLaVA** | **LLaMA-2-7B** | **72.72** | **757.16** | **30.9** | **9.2** |
| LLaVA | LLaMA-2-13B | 78.60 | 711.22 | 30.3 | 8.7 |
| **Causal-LLaVA** | **LLaMA-2-13B** | **79.54** | **872.09** | **28.2** | **8.5** |
| LLaVA1.5 | Vicuna-1.5-7B | 87.34 | 1508.51 | 52.1 | 14.9 |
| **Causal-LLaVA1.5** | **Vicuna-1.5-7B** | **88.18** | **1522.10** | **51.4** | **14.8** |

视觉理解能力同时提升：MMBench +2.0%, MM-Vet +4.8%, GQA +2.7%, VizWiz +8.4%。

### 消融实验要点
- **双路 vs 单路**：Only-projector (MME 748.70) + Only-transformer (726.15) < Both (757.16)，双路互补
- **投影矩阵选择**：共享 $W_k/W_v$ 最优（CHAIR_s 27.7），独立 $W_q/W_v$ 或 $W_q/W_o$ 导致灾难性退化（CHAIR_i 24.2-24.8）
- **PCA 可视化**：解纠缠后，原本紧密聚类的"餐桌"及其共现物体在所有层都显著分离

## 亮点
- **表示层面的因果分析**是核心贡献——首次可视化并量化了共现偏差在 MLLM 各层的传播过程
- 端到端架构级解决方案，不需要合成数据、外部模型或后处理
- 混淆因子字典是简洁优雅的设计——用 80 类物体的平均表示就足够
- 可视化分析极为充分（6 组 PCA 图覆盖原始/解纠缠 × 视觉/文本 × 多物体）

## 局限性 / 可改进方向
- 需 8×H20 GPU，计算资源要求较高
- 混淆因子估计可能受噪声或数据集分布影响
- 基于 LLaVA（较早的 MLLM），未在更新模型（如 InternVL、Qwen-VL）上验证
- CHAIR 提升幅度在 LLaVA 1.5 上较小（52.1→51.4），可能因 1.5 版本已部分缓解数据偏差

## 与相关工作的对比
- **vs VCD/OPERA（对比解码）**: 对比解码在推理时干预，不改变表示学习；Causal-LLaVA 从训练过程中解决根因
- **vs LRV/VIGC（数据校正）**: 数据方法依赖 GPT-4 生成，有错误传播风险；本方法不需要额外数据
- **vs Deconfounded Captioning**: 先前因果方法仅在输出 Softmax 层做 NWGM 近似；本文将因果干预深入到特征空间，在 projector 和 Transformer 层双路干预

## 启发与关联
- 共现偏差解纠缠的思路可用于场景图生成（Scene Graph Generation）的去偏差
- 混淆因子字典可动态更新——用在线估计替代离线统计，适应领域变化
- 与 BACL（同批次笔记）的关联：BACL 用模糊负样本改善对齐，Causal-LLaVA 用因果干预解偏差——两者可结合

## 评分
- 新颖性: ⭐⭐⭐⭐ 表示层面的因果分析视角新颖，但因果干预在 VQA/captioning 中有先例
- 实验充分度: ⭐⭐⭐⭐ 10 个基准、多 backbone、大量可视化，但缺乏最新 MLLM 对比
- 写作质量: ⭐⭐⭐⭐⭐ 分析驱动的研究范式优秀，从现象到原因到解决方案逻辑清晰
- 价值: ⭐⭐⭐⭐ 为 MLLM 幻觉提供了有因果理论支撑的架构级解决方案
