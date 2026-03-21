# EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation

**会议**: CVPR 2025  
**arXiv**: [2603.12108](https://arxiv.org/abs/2603.12108)  
**代码**: https://github.com/VisionXLab/EvoTok  
**领域**: 图像生成 / 多模态  
**关键词**: 统一图像Tokenizer, 残差量化, 视觉理解与生成, VQ-VAE, 多模态LLM

## 一句话总结
EvoTok 提出了一种基于残差潜在演化（Residual Latent Evolution）的统一图像 tokenizer，通过在共享潜空间中级联残差向量量化，使表示从浅层的像素级细节渐进演化到深层的语义级抽象，在仅用 13M 图像训练的情况下实现了 0.43 rFID 的重建质量，并在 7/9 个理解 benchmark 和 GenEval/GenAI-Bench 上取得优异效果。

## 研究背景与动机
1. **领域现状**：统一多模态 LLM 需要同时支持视觉理解（高层语义）和图像生成（像素级细节），核心问题是设计一个统一的图像 tokenizer。
2. **现有痛点**：现有两种范式各有缺陷——
   - **纠缠式**（ViLA-U, UniTok）：在同一组特征上同时施加 VQ 重建和对比学习 loss，导致语义对齐和像素重建之间的优化冲突
   - **解耦式**（DualToken, TokLIP, TokenFlow）：用独立分支/层/codebook 分别建模语义和像素特征，但过度独立导致两种特征之间缺乏内在一致性
3. **核心矛盾**：理解需要高层语义抽象，生成需要细粒度像素表示——两者似乎矛盾，但本质上视觉信息是一个从像素到语义的连续谱。
4. **本文要解决什么？** 如何在一个统一的潜空间中同时实现解耦（减少任务冲突）和一致性（共享视觉结构和语义先验）？
5. **切入角度**：将图像表示为残差量化的演化轨迹——浅层捕获像素细节，深层逐步积累为语义抽象，两者在同一空间中共同演化。
6. **核心idea一句话**：残差量化的不同深度自然对应像素→语义的演化谱，解耦来自不同深度切分，一致性来自共享空间。

## 方法详解

### 整体框架
输入图像 $I$ → 共享编码器 $\mathcal{E}$ 提取特征 $\mathbf{f}$ → $L$ 级残差向量量化 $\mathcal{RQ}$ 得到 $(\mathbf{k}_1, ..., \mathbf{k}_L)$ → 前 $L_{\text{pix}}$ 级的部分和为像素特征 $\mathbf{f}_{\text{pix}}$（送入像素解码器重建图像）→ 全部 $L_{\text{sem}}$ 级的累积和为语义特征 $\mathbf{f}_{\text{sem}}$（对齐 SigLIP2 语义特征，送入 LLM 做理解）。生成时 LLM 自回归预测 $L_{\text{pix}}$ 级 residual codes。

### 关键设计

1. **残差潜在演化（Residual Latent Evolution）**:
   - 做什么：用 RQ-VAE 的级联残差量化将图像编码为共享空间中的演化轨迹
   - 核心思路：$\mathbf{k}_i = \mathcal{Q}(\mathbf{r}_{i-1}; \mathcal{C}_i)$，$\mathbf{r}_i = \mathbf{r}_{i-1} - \mathbf{e}_i(\mathbf{k}_i)$。像素特征 $\mathbf{f}_{\text{pix}} = \sum_{i=1}^{L_{\text{pix}}} \mathbf{e}_i(\mathbf{k}_i)$，语义特征 $\mathbf{f}_{\text{sem}} = \sum_{i=1}^{L_{\text{sem}}} \mathbf{e}_i(\mathbf{k}_i)$
   - 设计动机：标准 RQ-VAE 中浅层自然捕获主体结构，深层捕获精细残差。EvoTok 反转这一用法——让浅层对应像素（前4级），全部16级对应语义，实现像素→语义的渐进演化

2. **像素→语义的方向选择**:
   - 做什么：验证了演化方向的重要性（pixel-to-semantic vs semantic-to-pixel）
   - 核心思路：消融实验表明 $(L_{\text{pix}}=4, L_{\text{sem}}=16)$ 优于 $(L_{\text{pix}}=16, L_{\text{sem}}=4)$ 和纠缠式 $(L_{\text{pix}}=L_{\text{sem}}=4)$
   - 设计动机：高层语义可以从像素特征逐步积累得到，但反向回归（从语义回到像素）效果不佳

3. **统一训练目标**:
   - 做什么：同时优化像素重建和语义对齐
   - 核心思路：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pix}} + \mathcal{L}_{\text{sem}} + \mathcal{L}_{\text{VQ}}$，其中 $\mathcal{L}_{\text{pix}}$ 包含重建+感知+对抗 loss，$\mathcal{L}_{\text{sem}}$ 为与 SigLIP2 特征的余弦相似度，$\mathcal{L}_{\text{VQ}}$ 为标准 VQ loss
   - 设计动机：两种 loss 作用在轨迹的不同深度切面上，自然解耦、不冲突

4. **理解与生成集成**:
   - 理解：$\mathbf{f}_{\text{sem}}$ 经语义解码器 + MLP 投影器送入 LLM（LLaVA 范式）
   - 生成：LLM 自回归预测空间位置后，RQ-Transformer head 沿深度预测 $L_{\text{pix}}$ 个 codes，累加得到像素特征后解码图像

### 损失函数
$$\mathcal{L}_{\text{total}} = \underbrace{(\mathcal{L}_R + \lambda_P \mathcal{L}_P + \lambda_G \mathcal{L}_G)}_{\text{pixel}} + \underbrace{\mathcal{L}_{\text{sem}}}_{\text{semantic}} + \underbrace{\mathcal{L}_{\text{VQ}}}_{\text{codebook}}$$

## 实验关键数据

### 主实验——理解（Unified Discrete 类）

| 方法 | LLM | SEEDBench | GQA | MMMU | MME |
|------|-----|-----------|-----|------|-----|
| VILA-U | LLaMA-7B | 59.0 | 60.8 | 33.5 | 1401.8 |
| DualToken | LLaMA-2-7B | 71.8 | — | 40.5 | 1502.7 |
| EMU3 | 8B | 68.2 | 60.3 | 37.2 | — |
| **EvoTok** | Qwen2.5-7B | **71.8** | **61.8** | **39.9** | **1895.1** |

### 主实验——生成

| 方法 | GenAI-Bench (Basic) ↑ | GenEval (Overall) ↑ | Position ↑ | Color Attri. ↑ |
|------|----------------------|---------------------|-----------|----------------|
| SDXL | 0.83 | 0.55 | 0.15 | 0.23 |
| EMU3 | — | 0.66 | 0.49 | 0.45 |
| **EvoTok** | **0.87** | **0.75** | **0.69** | **0.62** |

重建质量：rFID 0.43（ImageNet-1K 256×256），仅用 13M 训练数据。

### 消融实验

| 配置 ($L_{\text{pix}}, L_{\text{sem}}$) | rFID ↓ | SEEDBench ↑ | GenEval ↑ | 说明 |
|----------------------------------------|--------|-------------|-----------|------|
| (4, 4) Entangled | 0.66 | 62.7 | 0.64 | 纠缠式，两方面都差 |
| (16, 4) Sem→Pix | 0.44 | 64.6 | 0.60 | 重建好但生成/理解差 |
| **(4, 16) Pix→Sem** | **0.55** | **67.1** | **0.67** | **最均衡** |

### 关键发现
- 像素→语义方向（Pix→Sem）是唯一能在理解、生成、重建三个维度同时表现好的配置
- t-SNE 可视化清晰展示了从浅层到深层的连续演化轨迹：浅层聚类基于纹理/颜色，深层聚类对应语义类别
- CLIPSIMpix 在 depth 4 后进入平台期，而 CLIPSIMsem 持续上升——两者的功能分工在第 4 层处发生清晰切换
- 仅 13M 训练数据即达到强竞争力，说明架构设计本身的高效性

## 亮点与洞察
- **残差量化 = 自然的像素→语义谱**：核心洞察简洁而深刻——RQ-VAE 的级联残差本身就提供了不同粒度的信息层级，只需选择合适的深度切分点就能自然解耦像素和语义，无需额外分支
- **解耦 + 一致性的优雅统一**：两种特征在同一空间中共享轨迹前缀（前4级），又在后续层级分化，理论上实现了"局部共享、全局分工"
- **极少数据的强效果**：13M vs 数十亿级数据的竞品，说明好的归纳偏置比大数据更重要
- **可迁移思路**：residual evolution 的思想可以推广到音频（像素→语义）、视频（帧级→序列级语义）的统一 tokenizer

## 局限性 / 可改进方向
- 当前仅支持 256×256 分辨率，高分辨率（512, 1024）的扩展性待验证
- $L_{\text{pix}}=4$ 的像素特征可能不足以重建非常精细的纹理——rFID 0.43 虽好但与 VAE-based 方法仍有差距（0.55 vs 纯重建模型更低）
- 语义对齐依赖 SigLIP2，teacher 模型的选择对最终性能影响待探究
- 生成仍用标准 next-token prediction，未探索扩散解码等更强的生成范式

## 相关工作与启发
- **vs ViLA-U**: 纠缠式统一 tokenizer，同一特征空间做 VQ + 对比学习导致冲突。EvoTok 通过深度切分解耦，SEEDBench 71.8 vs 59.0
- **vs DualToken**: 层级式解耦（不同 encoder 层取特征），但缺乏共享空间。EvoTok 用残差轨迹的前缀共享保证一致性
- **vs TokenFlow**: 独立编码器式解耦，维护成本高。EvoTok 共享单一编码器更简洁
- **vs EMU3**: 8B from-scratch 一体化模型，数据和计算量大得多。EvoTok 用 7B 指令微调 LLM + 13M 数据达到竞争力

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 残差演化统一解耦和一致性的思路非常优雅
- 实验充分度: ⭐⭐⭐⭐ 理解/生成/重建三方面 + 消融 + 可视化分析
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰，可视化分析有洞察力
- 价值: ⭐⭐⭐⭐⭐ 统一 tokenizer 的新范式，简洁有效，影响力大
