# Directional Textual Inversion for Personalized Text-to-Image Generation

**会议**: ICLR 2026  
**arXiv**: [2512.13672](https://arxiv.org/abs/2512.13672)  
**代码**: [https://github.com/kunheek/dti](https://github.com/kunheek/dti)  
**领域**: 扩散模型 / 个性化生成  
**关键词**: Textual Inversion, 方向优化, 超球面, von Mises-Fisher, 个性化文本到图像  

## 一句话总结

本文发现 Textual Inversion (TI) 学到的 token embedding 存在范数膨胀（norm inflation）问题，导致复杂 prompt 的文本对齐下降；提出 Directional Textual Inversion (DTI)，将 embedding 范数固定在分布内尺度、仅在单位超球面上用 Riemannian SGD 优化方向，结合 von Mises-Fisher 先验，显著提升 prompt 忠实度。

## 研究背景与动机

1. **领域现状**：个性化文本到图像生成有两大范式——参数微调（如 DreamBooth）和嵌入优化（如 Textual Inversion）。TI 因为只优化 token embedding，具备存储小、易集成的优势，是很多后续方法的基础组件。
2. **现有痛点**：TI 在复杂 prompt 下表现很差——例如 "A painting of \<dog\> wearing a santa hat"，模型可能生成了狗但忽略了帽子和背景细节。根本原因是 TI 优化过程中 embedding 的范数会膨胀到极端值（>20，而正常词汇约 0.4）。
3. **核心矛盾**：语义信息主要编码在 embedding 的**方向**中（余弦相似度语义一致，欧几里得距离则不行），但 TI 不约束范数，导致：(a) 大范数在 pre-norm Transformer 中压制位置编码信息（$\mathcal{O}(1/m)$）；(b) 残差更新停滞，后续层无法有效修改 hidden state 方向。
4. **本文要解决什么？** 在保持 TI 轻量级优势的同时，解决 norm inflation 导致的文本对齐失败问题。
5. **切入角度**：作者从 CLIP token embedding 空间的几何结构出发，通过实验和理论两条线证明了"方向编码语义、范数膨胀有害"。这是一个可解释性驱动的分析视角。
6. **核心 idea 一句话**：固定 embedding 范数为分布内尺度，仅在单位超球面上优化方向，用 vMF 先验正则化。

## 方法详解

### 整体框架

DTI 将 token embedding $\bm{e} \in \mathbb{R}^d$ 解耦为范数 $m^\star$ 和方向 $\bm{v} \in \mathbb{S}^{d-1}$，即 $\bm{e} = m^\star \bm{v}$。范数固定为预训练词汇表 embedding 的均值范数，只优化方向 $\bm{v}$。优化在单位超球面上进行，使用 Riemannian SGD，并加入 von Mises-Fisher (vMF) 方向先验作为正则化。

### 关键设计

1. **超球面方向优化（Riemannian SGD）**:
   - 做什么：在 $\mathbb{S}^{d-1}$ 上优化 embedding 方向，避免范数膨胀
   - 核心思路：先将欧几里得梯度投影到切空间 $\bm{g} = \bm{g}_{\text{euc}} - (\bm{v}_k^\top \bm{g}_{\text{euc}})\bm{v}_k$，再通过 retraction 映射回球面 $\bm{v}_{k+1} = \frac{\bm{v}_k - \eta \bm{g}}{\|\bm{v}_k - \eta \bm{g}\|}$。此外还对梯度做了归一化 $\bm{g}' = \bm{g}/\|\bm{g}\|$
   - 设计动机：欧几里得空间的 AdamW 会让参数漂离流形，不适合球面约束。RSGD 尊重流形几何，消融实验证实优于 AdamW + 投影

2. **von Mises-Fisher (vMF) 方向先验**:
   - 做什么：将方向优化视为 MAP 估计，引入 vMF 分布作为先验，防止语义漂移
   - 核心思路：$p(\bm{v}|\bm{\mu}, \kappa) \propto \exp(\kappa \bm{\mu}^\top \bm{v})$，其中 $\bm{\mu}$ 是对应类别词（如 'dog'）的归一化 embedding。负对数先验梯度为常量 $-\kappa\bm{\mu}$，直接加到数据梯度上即可
   - 设计动机：类似去耦权重衰减的思想，但适配到球面；$\kappa$ 固定为 1e-4，计算开销极小

3. **范数尺度选择**:
   - 做什么：将 $m^\star$ 固定为预训练词汇表 embedding 的均值范数
   - 设计动机：消融实验表明，用最小值范数会导致主体相似度崩塌，用 OOD 大范数则文本对齐变差，均值最佳

### 损失函数 / 训练策略

数据损失为标准扩散去噪 MSE：$\mathcal{L}_{\text{data}}(m^\star \bm{v}) = \mathbb{E}[\|\bm{\epsilon} - \bm{\epsilon}_\theta(\bm{z}_t, t, c(m^\star \bm{v}))\|^2]$。先验损失 $\mathcal{L}_{\text{prior}} = -\kappa \bm{\mu}^\top \bm{v}$，总损失为两者之和。训练约 7 分钟/概念（SDXL，单卡 A6000）。

## 实验关键数据

### 主实验

| 模型 | 方法 | Image Sim (DINOv2) | Text Sim (SigLIP) |
|------|------|------|------|
| SDXL | TI | 0.561 | 0.292 |
| SDXL | TI-rescaled | 0.243 | 0.466 |
| SDXL | CrossInit | 0.545 | 0.464 |
| SDXL | **DTI (ours)** | **0.450** | **0.522** |
| SANA 1.5-1.6B | TI | 0.480 | 0.621 |
| SANA 1.5-1.6B | **DTI (ours)** | **0.479** | **0.744** |
| SANA 1.5-4.8B | TI | 0.446 | 0.646 |
| SANA 1.5-4.8B | **DTI (ours)** | **0.452** | **0.757** |

DTI 在所有模型上大幅提升文本对齐（SDXL上 0.292→0.522），同时保持合理的主体相似度。随模型增大优势更显著。

### 消融实验

| 优化器 | $m^\star$ | $\kappa \times 10^{-3}$ | Image | Text |
|--------|---------|---------|-------|------|
| AdamW | mean | 0.1 | 0.335 | 0.463 |
| RSGD | min | 0.1 | 0.030 | 0.074 |
| RSGD | 5.0 (OOD) | 0.1 | 0.383 | 0.373 |
| RSGD | mean | 0.0 | 0.507 | 0.436 |
| RSGD | mean | 0.5 | 0.278 | 0.688 |
| **RSGD** | **mean** | **0.1** | **0.450** | **0.522** |

### 关键发现

- RSGD 显著优于 AdamW+投影，说明尊重流形几何很重要
- 范数设为最小值或 OOD 值效果极差，均值最优
- vMF 先验不可或缺（$\kappa=0$ 时文本对齐明显下降），但 $\kappa$ 过大也会损害图像相似度
- 用户研究（100 人 AMT）中 DTI 在主体忠实度（43.45%）和文本对齐（66.77%）均排第一

## 亮点与洞察

- **理论分析扎实**：从 pre-norm Transformer 的数学结构出发，证明了 norm inflation → 位置信息衰减 + 残差更新停滞的因果链（Proposition 1, Corollary 1），这是对 TI 失败模式的首个系统性理论解释
- **球面插值 (SLERP) 能力**：DTI 的超球面参数化天然支持学习到的概念之间的平滑语义插值（如狗↔茶壶、猫↔狗），这是标准 TI 做不到的。这一能力开拓了概念混合的创意应用
- **极简且高效**：整个方法相比 TI 只改了优化过程——固定范数 + RSGD + 常量先验梯度，无额外网络、无额外存储，训练时间不增加

## 局限性 / 可改进方向

- DTI 主要改善文本忠实度，并不直接优化主体相似度；高主体保真度需要搭配 LoRA 等方法
- 理论分析聚焦 pre-norm 架构（CLIP, Gemma），对 post-norm 或其他归一化方案是否适用未知
- vMF 先验的 $\kappa$ 需要手动设定，虽然论文说 1e-4 通用，但不同概念复杂度下可能需要调整
- 仍然需要每个概念单独训练（SDXL ~7min），无法做到 zero-shot 个性化

## 相关工作与启发

- **vs TI**: TI 不约束范数导致 embedding OOD，DTI 通过固定范数+方向优化根本解决了这个问题
- **vs CrossInit**: CrossInit 在 SDXL 上文本对齐不错但在 SANA（LLM-based encoder）上失效，DTI 跨架构泛化更好
- **vs P+/NeTI**: 这些方法通过更丰富的 embedding 空间改善 TI，但引入大量计算开销，DTI 保持了 TI 的轻量优势
- 方向优化 + vMF 先验的思路可以迁移到 VLM prompt tuning 或 LLM soft prompt 优化中

## 评分

- 新颖性: ⭐⭐⭐⭐ 从几何视角解释 TI 的失败并给出简洁方案，洞察深刻
- 实验充分度: ⭐⭐⭐⭐ 多模型（SDXL/SANA）、消融完整、用户研究、插值实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论-实验-方法逻辑链非常清晰，图表精美
- 价值: ⭐⭐⭐⭐ 实用价值高，即插即用，对 TI 生态有广泛影响
