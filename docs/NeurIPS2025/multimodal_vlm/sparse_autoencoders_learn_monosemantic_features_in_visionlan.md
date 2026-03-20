# Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2504.02821](https://arxiv.org/abs/2504.02821)  
**代码**: [https://github.com/ExplainableML/sae-for-vlm](https://github.com/ExplainableML/sae-for-vlm)  
**领域**: 多模态VLM / 可解释性 / AI安全  
**关键词**: sparse autoencoder, monosemanticity, CLIP, VLM interpretability, model steering  

## 一句话总结
将Sparse Autoencoder (SAE)从LLM可解释性扩展到VLM领域，提出MonoSemanticity Score (MS)量化视觉神经元的单义性，发现SAE能将VLM中多义的神经元分解为单义特征，且可直接通过操控单个SAE神经元来steering LLaVA的输出（插入或抑制概念），无需修改LLM。

## 背景与动机
SAE已成为LLM可解释性的重要工具（如Anthropic的Claude解释工作），但在VLM领域的应用有限。VLM（如CLIP）的神经元天然是多义的（polysemantic）——一个神经元同时响应手机和尺子。缺少一个量化"单义性"的指标来评估SAE在视觉域的效果，也缺少将SAE用于VLM控制的实践。

## 核心问题
SAE能否有效分解VLM的视觉表示为单义特征？如何量化评估？分解后的特征能否用于无监督地steering多模态LLM的输出？

## 方法详解

### 整体框架
三部分：(1)定义MonoSemanticity Score (MS)评估SAE神经元单义性；(2)系统比较不同SAE架构/超参数对MS的影响；(3)将CLIP上训练的SAE迁移到LLaVA做概念级steering。

### 关键设计

1. **MonoSemanticity Score (MS)**：对每个SAE神经元k，在大规模图像集上计算激活加权的成对图像相似度：MS_k = Σ(r_nm^k · s_nm) / Σ(r_nm^k)，其中r是激活相关矩阵（两张图共同激活该神经元的程度），s是DINOv2嵌入的余弦相似度。分数越高→该神经元的高激活图像越相似→越单义。通过1000人次的大规模用户研究验证：MS差异>0.1时人类alignment rate从56.6%升至100%。

2. **SAE架构对比**：比较BatchTopK和Matryoshka BatchTopK SAE。关键发现：
   - Matryoshka SAE整体MS更高（但R²低2-3%）
   - 宽latent（expansion factor↑）→最高MS神经元更单义
   - 稀疏度↑（K↓）→整体MS↑（但K=1时R²仅31%，K=20的66.8%是好的平衡）
   - 即使expansion factor=1（与原始层同宽），90%的SAE神经元比原始神经元更单义→稀疏重构本身就促进概念分离

3. **Steering MLLM**：在CLIP vision encoder的layer 22上训练SAE，插入LLaVA的vision encoder后。操控方式：将SAE neuron k的激活设为常数α（正值→插入概念，负值→抑制概念），其他neuron不变，然后用SAE decoder重构回token embedding。
   - 概念插入：用白图+文本prompt，增大"铅笔neuron"的α→输出从love poem逐渐变成关于铅笔的诗
   - 概念抑制：看有刀和草莓的图，减小"刀neuron"的α→输出描述中刀逐渐消失，草莓保留

### 评估
在CLIP ViT-L/14、SigLIP、AIMv2、WebSSL等4个vision encoder上验证MS，在LLaVA-1.5-7b上验证steering。

## 实验关键数据

| 设置 | 最高MS | 平均MS |
|------|--------|--------|
| 原始CLIP神经元 (无SAE) | 0.01 | 0.01 |
| BatchTopK SAE ε=4 | 0.80 | 0.20 |
| Matryoshka SAE ε=4 | 0.87 | 0.23 |
| Matryoshka SAE ε=64 | 1.00 | ~0.18 |

Steering量化对比 (SAE vs Difference-in-Means)：
- 概念插入：SAE 42.4% vs DiffMean 35.8%（双标准满足率）
- 概念抑制：SAE 52.5% vs DiffMean 33.3%
- SAE在保持base prompt跟随方面远超DiffMean（85.8% vs 66.2%）

### 消融实验要点
- 稀疏性是关键：K=1→MS最高但R²太低；K=20是好平衡
- expansion factor > 4后MS的相对分布开始下降（虽然绝对数量仍增加）
- 跨模型泛化：SAE从CLIP迁移到SigLIP/AIMv2/WebSSL都有效
- Matryoshka层次与iNaturalist分类树的LCA深度相关→层次化概念发现

## 亮点 / 我学到了什么
- **MS指标简洁而有效**——激活加权成对相似度，82.8%人类对齐率，可作为SAE评估的标准工具
- **Vision SAE → MLLM steering**的迁移路径极其优雅——只改vision encoder的后处理，完全不碰LLM
- 概念抑制的potential：可用于过滤有害/不期望的视觉概念，在信息到达LLM之前就拦截
- expansion factor=1时就有显著提升→稀疏字典学习本身（而非仅仅增加维度）是概念分离的关键

## 局限性 / 可改进方向
- 高MS neuron并不总是精确的steering工具（如golden retriever neuron可能触发任何狗相关输出）
- 部分SAE neurons是feature detector不产生steering效果
- 仅在CLIP系vision encoder上验证，未测试SigLIP-based LLaVA或InternViT
- MS指标使用外部image encoder（DINOv2），引入了该encoder的bias
- 仅做image-level MS，未扩展到text domain

## 与相关工作的对比
- vs **Anthropic的SAE for Claude**：本工作将SAE从LLM可解释性扩展到VLM视觉域，MS指标是视觉版的评估方案
- vs **CLIP-Dissect/CLIP Decomposition**：这些工作用文本描述解释CLIP neuron，本工作用SAE解耦后再评估/steering
- vs **VL-SAE (2510.21323)**：VL-SAE关注VLM内部alignment的统一概念集，本工作更关注量化评估和下游steering

## 与我的研究方向的关联
- 与Narrow Gate (2412.06646)互补——Narrow Gate揭示native VLM的[EOI] token bottleneck，SAE可以进一步分解[EOI]的表示看具体编码了哪些概念
- 对VLM安全有直接意义——可以在vision encoder层面做concept-level filtering
- MS score可用于评估VLM token压缩方法是否损失了关键概念（压缩前后MS对比）

## 评分
- 新颖性: ⭐⭐⭐⭐ SAE用于VLM不是第一次，但MS指标+大规模用户研究+MLLM steering的组合是新的
- 实验充分度: ⭐⭐⭐⭐⭐ 4个vision encoder、多层多expansion factor、1000人次用户研究、定量steering评估
- 写作质量: ⭐⭐⭐⭐ 清晰系统，但部分表格信息偏密
- 对我的价值: ⭐⭐⭐⭐ VLM可解释性和安全对齐的实用工具，MS指标可直接复用
