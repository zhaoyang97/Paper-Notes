<!-- 由 src/gen_stubs.py 自动生成 -->
# Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT

**会议**: NeurIPS 2025  
**arXiv**: [2510.00296](https://arxiv.org/abs/2510.00296)  
**代码**: [https://github.com/BarSGuy/ACT-ViT](https://github.com/BarSGuy/ACT-ViT)  
**领域**: LLM/NLP  
**关键词**: 幻觉检测, 激活张量, Vision Transformer, 跨模型泛化, Probing

## 一句话总结
将LLM的全部隐层激活组织为"激活张量"（层×token×隐维度），类比图像用ViT处理，设计ACT-ViT架构支持跨LLM联合训练，在15个LLM-数据集组合上一致超越传统probing方法，并展现出对未见数据集和未见LLM的强零样本/少样本迁移能力。

## 研究背景与动机
1. **领域现状**：检测LLM幻觉的方法中，probing分类器（在隐层表征上训练线性分类器）是高效的白盒方法。但传统probing在孤立的单层-单token位置上操作，需要预先确定最佳层和token位置。

2. **现有痛点**：
   - **信号位置不固定**：最佳probing位置在不同样本、不同数据集、不同LLM之间变化很大——Mistral的最佳位置是(第14层, token 0)，而Qwen的最佳位置在最后几层的末尾token
   - **LLM特异性**：每个LLM都需要单独训练探针，无法跨模型共享数据集或迁移学习
   - **不完整利用**：只用一个层-token位置的激活，浪费了大量信息

3. **核心洞察**：激活张量 $\mathbf{A} \in \mathbb{R}^{L \times N \times D}$（层数×token数×隐维度）在结构上类似于图像（高×宽×通道），可以借用视觉模型的方法来处理。

4. **核心idea一句话**：把LLM的全部隐层激活当作"图像"，用ViT自适应地attend到最有信息量的层-token组合，实现跨LLM的高效幻觉检测。

## 方法详解

### 整体框架
提取LLM的激活张量 → Pooling压缩空间维度(层和token方向) → 每个LLM用专属的Linear Adapter映射到共享特征空间 → 共享的ViT Backbone处理 → 二分类(幻觉/正确)。

### 关键设计

1. **激活张量（Activation Tensor）**：
   - 定义：$\mathbf{A} \in \mathbb{R}^{L_M \times N \times D_M}$，包含LLM所有层在所有输出token上的隐层状态
   - 与图像的类比：层→垂直空间维度，token→水平空间维度，隐维度→通道
   - 包含了完整的内部状态信息，避免了选择特定层/token的信息损失

2. **Pooling + Linear Adapter**：
   - Pooling：对"空间"维度（层和token）做max-pooling，统一为固定大小 $(L_p, N_p) = (8, 100)$，解决不同LLM层数不同、不同输入token数不同的问题
   - Linear Adapter：每个LLM $M$ 有独立的线性变换 $\mathbf{W}_M \in \mathbb{R}^{D_M \times D'}$，将不同隐维度映射到共享维度 $D'$
   - 设计动机：受"不同LLM学习了近似线性可转换的真实世界表征"这一假设驱动。单个线性层足以对齐不同LLM的特征空间

3. **ViT-Based Backbone**：
   - 将pooled+adapted的张量切成不重叠的patch，添加patch内位置编码+全局位置编码
   - 展平patch后通过标准Transformer编码器
   - 自注意力机制让模型自适应地attend到最有幻觉信号的层-token位置，无需预先指定

### 训练策略
- 联合训练：在所有可用LLM和数据集上同时训练，共享ViT backbone，各LLM独立LA
- 对新LLM的迁移：冻结backbone，只训练新LLM的LA（轻量级适配）
- 在单GPU上3小时内训练完全部15个组合，推理速度 $\approx 10^{-5}$ 秒/样本

## 实验关键数据

### 主实验（AUC，15个LLM-数据集组合）

| 方法 | Mis-7B Movies | LlaMa-8B TriviaQA | Qwen-7B HQA | 平均提升 |
|------|---------------|-------------------|-------------|----------|
| Logits-mean | 63.0 | 66.0 | 66.2 | - |
| Probe[*] (最佳层-token) | ~80-85 | ~75-82 | ~72-80 | - |
| ACT-ViT(s) (单组合) | ~85-88 | ~80-84 | ~78-83 | +3-5 vs Probe |
| **ACT-ViT** (多LLM联合) | **~88-92** | **~84-88** | **~82-87** | **+5-10 vs Probe** |

### 迁移学习

| 设置 | 效果 |
|------|------|
| 零样本到新数据集（已见LLM） | 强泛化，很多情况超过在目标数据集上训练的Probe |
| 5%数据微调LA到新LLM | 在多数情况下超过在100%数据上训练的单模型Probe |
| 多LLM联合 vs 单LLM | 联合训练一致更好，跨LLM知识确实互补 |

### 关键发现
- ACT-ViT在15个组合中一致超越传统probing，平均提升显著
- 多LLM联合训练显著优于单模型训练——不同LLM的幻觉信号可以互补
- 对新LLM只需训练LA（参数极少），5%数据就够——实际部署场景非常友好
- 零样本对新数据集泛化也很强，说明幻觉检测信号有跨任务共性
- ViT的自注意力比MLP更有效——ACT-MLP（flatten后用MLP）性能明显更差

## 亮点与洞察
- **"激活张量=图像"的类比**非常优雅：把一个NLP问题转化为视觉问题，借用ViT的自注意力机制自适应地找到最有信号的层-token位置，完全避免了传统probing需要预先选位置的难题
- **跨LLM联合训练**的成功验证了一个重要假设：不同LLM编码幻觉的方式存在共性，可以通过线性变换对齐
- **极致的效率**：推理$10^{-5}$秒/样本（比LLM-based检测方法快5个数量级），训练3小时搞定15个组合

## 局限性 / 可改进方向
- 需要白盒访问LLM的所有层隐状态——对API-only模型不适用
- 激活张量的存储开销大（单个LLM约0.2GB/样本），大规模部署需要优化存储
- 只测试了7-8B规模的模型，对更大（70B+）或更小（1B）模型的效果未知
- 线性适配假设可能在架构差异很大的LLM之间不成立
- 只关注事实性QA类幻觉——对推理错误、主观偏见等更复杂的错误类型效果未知

## 相关工作与启发
- **vs Orgad et al. (2024)**：他们发现"exact token"probing的重要性但仍需外部算法定位。ACT-ViT通过处理完整激活张量自动解决了定位问题
- **vs logits/概率方法**：不需训练但信息有限（只用输出层）。ACT-ViT利用所有层的信息
- **对可解释性的启示**：ViT的attention map可以揭示幻觉信号主要来自哪些层-token组合，为LLM可解释性提供新视角

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 激活张量类比图像的视角和跨LLM联合训练范式都是全新的
- 实验充分度: ⭐⭐⭐⭐⭐ 15个组合、多种设置（单模型/多模型/零样本/少样本/迁移）、完整消融
- 写作质量: ⭐⭐⭐⭐⭐ 类比直观，Figure 1设计精美，实验分析系统性强
- 价值: ⭐⭐⭐⭐⭐ 为幻觉检测提供了高效通用的新范式，跨LLM迁移能力是重要突破
