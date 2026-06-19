---
title: >-
  [论文解读] RNN as Linear Transformer: A Closer Investigation into Representational Potentials of Visual Mamba Models
description: >-
  [CVPR 2026][Vision Mamba] 本文把 Softmax 注意力、线性注意力、Mamba 统一写成同一个 token-mixing 矩阵 $Y=MX$，用秩分析证明 Mamba 是 Softmax 注意力的"低秩近似"、表达力严格夹在二者之间，并提出 Binary-AUC 指标把特征图质量从"肉眼看"变成可量化的 AUC，最终用 DINO 自监督训出的 Vision Mamba 在 ImageNet 线性探测达到 78.5%。
tags:
  - "CVPR 2026"
  - "Vision Mamba"
  - "秩分析"
  - "线性注意力"
  - "自监督DINO"
  - "特征图评估"
---

# RNN as Linear Transformer: A Closer Investigation into Representational Potentials of Visual Mamba Models

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_RNN_as_Linear_Transformer_A_Closer_Investigation_into_Representational_Potentials_CVPR_2026_paper.html)  
**代码**: https://github.com/yangtiming/Dino-Mamba  
**领域**: 表征理论 / Vision Mamba  
**关键词**: Vision Mamba, 秩分析, 线性注意力, 自监督DINO, 特征图评估

## 一句话总结
本文把 Softmax 注意力、线性注意力、Mamba 统一写成同一个 token-mixing 矩阵 $Y=MX$，用秩分析证明 Mamba 是 Softmax 注意力的"低秩近似"、表达力严格夹在二者之间，并提出 Binary-AUC 指标把特征图质量从"肉眼看"变成可量化的 AUC，最终用 DINO 自监督训出的 Vision Mamba 在 ImageNet 线性探测达到 78.5%。

## 研究背景与动机

**领域现状**：Transformer 凭全局建模能力从 NLP 横扫视觉，但其 $O(L^2)$ 复杂度在高分辨率图像上难以扩展。Mamba 这类状态空间模型（SSM）以线性复杂度捕捉长程依赖，配合选择性扫描和硬件友好实现，被大量搬到视觉骨干、分割检测、医学图像等任务上，成为 Transformer 的热门替代。

**现有痛点**：Mamba 在视觉里"为什么work"几乎是个黑箱。已有工作（如 MILA、RALA）要么把 Mamba 解释成线性注意力的一个变体、归功于遗忘门，要么从门控机制角度切入——本质都是"把 Mamba 往注意力框架上靠"，没有从 Mamba 自身结构出发去严格界定它的表达能力到底处在什么位置。

**核心矛盾**：线性注意力虽然便宜，但用核函数替掉 softmax 后，注意力矩阵的秩被压到投影维度 $D_{QK}$，表达力大幅缩水；Softmax 注意力表达力满秩但要付 $O(L^2)$ 代价。Mamba 结构上像线性注意力（都是 $Y=MX$ 形式），凭什么效果明显更好？这个"表达力—效率"权衡里 Mamba 究竟落在哪？

**本文目标**：(1) 给三种 token mixer 一个统一的数学框架并比较秩上界；(2) 把"特征图好不好"做成可量化指标；(3) 验证 Mamba 在自监督范式下的实际表征潜力。

**切入角度**：作者注意到 Mamba-2 的 SSD 形式可以写成一个半可分矩阵 $M$，而 Softmax/线性注意力也都能写成 $Y=MX$。既然三者共用同一个壳子，差别就全压在 $M$ 的结构上——于是用矩阵秩这一把统一的尺子去量它们的表达力。

**核心 idea**：把 Mamba 看成"带可学习因果掩码的线性注意力"，证明其可学习掩码 $L_M$ 把 off-diagonal 块的秩从线性注意力的 $D_{QK}$ 抬高到 $\mathrm{rank}(L_M)\cdot N$，从而在秩层面建立 Softmax > Mamba > 线性注意力 的严格层级，再用一个基于分割掩码的 AUC 指标把这条理论结论在真实特征图上量出来。

## 方法详解

本文不是提出一个新网络，而是"分析 + 度量 + 验证"三件事：先用统一框架做秩分析得到一条表达力层级链，再设计 Binary-AUC 把特征图质量量化，最后用 DINO 自监督把 Vision Mamba 训出干净特征图来佐证理论。因为核心是矩阵层面的机制分析、不存在多阶段串行 pipeline，这里不画框架图，用公式把等价关系讲清。

### 整体框架

第一步是**统一形式化**：给定输入 $X\in\mathbb{R}^{L\times d}$，三种 token mixer 的输出都写成 $Y=MX$，区别只在混合矩阵 $M$ 怎么构造。第二步是**秩分析**：把 $M$ 按 chunk 切成对角块（块内）和非对角块（跨块长程依赖），证明对角块三者都满秩、差异全在非对角块的秩上界，由此排出 Softmax > Mamba > 线性注意力。第三步是**实证度量**：用 [CLS] token 与图像 token 的相似度图（可看作一次线性变换，其有效秩反映区分能力）配合分割掩码算 AUC，再用 DINO 自监督训练得到更干净的特征图，把理论层级在分类/分割/检测/鲁棒性上逐一验证。

### 关键设计

**1. 三种 token mixer 的统一矩阵形式：把 Mamba 拉进同一张桌子比较**

要比较表达力，先得让三者可比。作者证明 Softmax 注意力、线性注意力、Mamba 都能写成 $Y=MX$，只是混合矩阵 $M$ 不同：

$$M=\begin{cases} L_M\circ(C^\top B), & \text{Mamba}\\ L_{\text{Attn}}\circ\mathrm{softmax}(QK^\top), & \text{Self-Attn}\\ L_{\text{Attn}}\circ\big(\phi(Q)\phi(K)^\top\big), & \text{Lin-Attn}\end{cases}$$

其中 $\circ$ 是逐元素（Hadamard）乘。关键在因果掩码这一项：Softmax 和线性注意力用的是**固定**的下三角全 1 掩码 $L_{\text{Attn}}$；而 Mamba 来自 SSM 递推 $h_t=A_t h_{t-1}+B_t x_t,\ y_t=C_t^\top h_t$，展开后 $M_{ij}=L_{M,ij}\circ(C_i^\top B_j)$、$L_{M,ij}=A_i\cdots A_{j+1}$，这里的 $L_M$ 是由状态转移 $A$ 连乘出来的**可学习、数据相关**的掩码。这一统一形式是后面秩分析的地基——它点明 Mamba 与线性注意力的本质区别不在 $C^\top B$ vs $\phi(Q)\phi(K)^\top$，而在"固定掩码 vs 可学习掩码"。

**2. 基于 Hadamard 秩界的表达力层级：证明 Mamba 严格夹在中间**

有了统一形式，作者用一条标准结论 Hadamard 秩界 $\mathrm{rank}(A\circ B)\le\mathrm{rank}(A)\cdot\mathrm{rank}(B)$ 来卡每种混合矩阵的秩。把 $M$ 切成 $C\times C$ 的子块：对角块因下三角结构三者都满秩 $R_{\text{diag}}=C$；差异在非对角块。Softmax 因 softmax 的非线性把固定掩码 $L_{\text{Attn}}$ 变成"等效可学习"，非对角块满秩 $R^{\text{off}}_{\text{Self}}=C$。线性注意力里固定掩码 $\mathrm{rank}(L_{\text{Attn}})=1$、核积秩受限于 $D_{QK}$，故 $R^{\text{off}}_{\text{Lin}}\le D_{QK}$。Mamba 的可学习掩码 $\mathrm{rank}(L_M)\ge1$、$C^\top B$ 秩受限于状态维 $N$，故 $R^{\text{off}}_{\text{Mamba}}\le\mathrm{rank}(L_M)\cdot N$。于是得到层级：

$$\underbrace{C}_{\text{Self-Attn}} > \underbrace{\mathrm{rank}(L_M)\cdot N}_{\text{Mamba}} > \underbrace{D_{QK}}_{\text{Lin-Attn}}$$

直觉就是：线性注意力被"固定掩码 + 低秩核积"双重限死秩；Mamba 把固定掩码换成可学习掩码、又能让状态维 $N$ 随复杂度 $O(LNJ)$ 高效扩展（而线性注意力的 $D_{QK}$ 受 $O(LD_{QK}^2)$ 二次成本拖累涨不动），两点叠加把秩抬到线性注意力之上、Softmax 之下。作者据此称 Mamba 是 Softmax 注意力的"低秩近似"，是效率与表达力之间一个有吸引力的中间地带。⚠️ 论文称"softmax 让 $L_{\text{Attn}}$ 从固定变可学习"是一种解释性论证，严格证明在附录，以原文为准。

**3. Binary-AUC：把"特征图好不好看"变成可量化的 AUC**

理论排出层级后需要在真实特征图上验证，但已有评估几乎全靠肉眼看 saliency map，没有客观指标。作者提出 Binary-AUC：把多类分割标签并成前景/背景二值掩码 $\text{Mask}_{\text{label}}$，对特征图按阈值 $t\in[0,1]$ 二值化得 $\text{Mask}^t_{\text{feature}}$，逐阈值算覆盖率 $R(t,S)=\frac{|\text{Mask}^t_{\text{feature}}\cap S|}{|S|}$，由此得到 TPR/FPR 曲线并积分出 AUC：

$$\text{AUC}=\sum_i (\text{FPR}_{i+1}-\text{FPR}_i)\cdot\frac{\text{TPR}_{i+1}+\text{TPR}_i}{2},\quad \text{AUC}_{\text{norm}}=\max(\text{AUC},1-\text{AUC})$$

AUC=1 表示特征图与真值前景完美对齐、0.5 是随机。这个指标第一次在 ImageNet 规模上用 [CLS]-token 相似度图客观评估特征质量，把"自注意力特征最干净、Mamba 次之、线性注意力最差"这种主观印象坐实成数字，并能拆到逐注意力头、逐层、register token 位置等粒度做诊断。

### 训练策略

模型用 DINO 自监督范式在 ImageNet-1k 预训练：输入图生成全局/局部 crop 喂学生网络 $S$ 和教师网络 $T$，教师参数由学生 EMA 更新，对教师输出中心化 + 温度 softmax 得 $P_t$（停梯度）、学生得 $P_s$，最小化交叉熵 $-P_t\log P_s$。骨干用 Mamba-v2（tiny/small/base 维度 256/512/768，均 24 层双向扫描），并把 Vim 的固定位置编码换成 DINO 的自适应位置编码（双三次插值）以兼容多尺度输入。线性注意力对照组 LinearViT 直接把 $\mathrm{softmax}(QK^\top)$ 换成 $\phi(Q)\phi(K)^\top$（$\phi$ 取逐行 softmax）。

## 实验关键数据

### 主实验

ImageNet-1k 线性探测（均为 DINO 预训练），自注意力 > Mamba > 线性注意力，与秩层级一致；Mamba 接近 ViT 但需要更多参数：

| 骨干 | Mixer | #Param.(M) | Top-1 (%) |
|------|-------|------------|-----------|
| ViT-B | self-attn | 85 | 78.2 |
| LinearViT-B | linear attn | 85 | 74.7 |
| DinoVim-B | mamba-2 | 88 | 78.1 |
| DinoMa.-R.-B（重探测头） | mamba-2 | 88 | **78.5** |

ADE20K 语义分割（UperNet）+ COCO 检测/实例分割（Cascade Mask R-CNN），高分辨率长序列任务上 Mamba 对线性注意力的优势被放大：

| 骨干 | ADE20K mIoU(%) | COCO APᵇ | COCO APᵐ |
|------|----------------|----------|----------|
| LinearViT-B | 29.2 | 37.1 | 32.6 |
| DinoVim-B | 38.0 | 42.8 | 37.4 |
| ViT-B | 43.2 | 44.8 | 39.1 |

### 鲁棒性 / 分析实验

ImageNet 分布偏移变体上，高秩架构泛化更好，Mamba 在线性复杂度下仍保持与 ViT 接近的鲁棒性：

| 骨干 | Sketch | ImageNet-A | ImageNet-R | Real |
|------|--------|-----------|-----------|------|
| LinearViT-B | 21.6 | 9.6 | 32.7 | 81.3 |
| DinoVim-B | 27.6 | 14.2 | 33.1 | 84.3 |
| ViT-B | 25.5 | 15.4 | 38.0 | 84.6 |

特征图质量（Binary-AUC）诊断分析：自监督 AUC 一致高于监督；模型越大特征图越干净（但 DINOv2 因大模型高范数离群点反而下降）；逐层看早期层 AUC 低于 50%（关注背景）、随深度上升转向目标特征；register token 放中间位置 AUC 最高，且 AUC 与线性探测精度强相关，呈相同的波动模式。

### 关键发现
- 三个任务上 Self-Attn > Mamba > Linear-Attn 的排序在分类/分割/检测/鲁棒性上**全部成立**，与秩层级 $C > \mathrm{rank}(L_M)\cdot N > D_{QK}$ 一一对应——理论预测被实证坐实。
- Mamba 相对线性注意力的优势在高分辨率长序列任务（分割 +8.8 mIoU、检测 +5.7 APᵇ）远大于分类（线性探测仅约 +3.4），印证其长程建模能力正是来自更高的秩。
- Binary-AUC 与线性探测精度强相关，说明特征图质量可作为模型性能的廉价代理指标。

## 亮点与洞察
- **统一壳子 + 一把尺子**：把三种看似不同的 token mixer 统一成 $Y=MX$、再用矩阵秩这一把尺子量，是非常清爽的分析框架，比"把 Mamba 解释成某种注意力变体"更本质——差别被精准定位到"固定掩码 vs 可学习掩码 + 状态维可扩展"。
- **可学习掩码是 Mamba 的秩来源**：最"啊哈"的一点是 Mamba 相对线性注意力的增益不在 value 投影、而在 $L_M$ 这个由状态转移连乘出来的数据相关掩码把 off-diagonal 秩抬了上去；这给"为什么 SSM 比线性注意力强"一个干净的代数解释。
- **把定性评估做成定量指标**：Binary-AUC 用现成分割标签 + ROC/AUC 把"特征图干不干净"量化，思路可直接迁移到任何骨干的可解释性诊断、甚至当作 NAS/训练的轻量代理信号。

## 局限与展望
- 秩分析给的是**上界**而非紧界，"$\mathrm{rank}(L_M)\cdot N > D_{QK}$ 在典型配置下成立"依赖 $C{=}256,N{=}D_{QK}{=}64$ 这类具体取值，换配置时层级是否仍严格成立需谨慎（作者已强调这是"典型部署"下的结论）。
- Binary-AUC 依赖分割真值掩码，只适用于有像素级标注的数据；且把多类标签并成前景/背景，丢掉了细粒度语义结构信息。
- 全程在 ImageNet 规模、24 层固定深度、双向扫描这一组设定下验证，未覆盖更大规模、不同扫描策略或非 DINO 自监督范式，结论的外推性有待进一步检验。

## 相关工作与启发
- **vs MILA**：MILA 把 Mamba 链接到线性注意力、把成功归因于遗忘门，本质是"把 Mamba 往注意力靠"；本文反过来从 Mamba 自身的半可分矩阵出发建立严格秩层级，解释更代数化、更能区分三者表达力。
- **vs RALA**：RALA 针对线性注意力的低秩限制做"秩增强"；本文不改架构，而是用秩分析说明 Mamba 凭可学习掩码天然就比线性注意力高秩，提供了"为什么需要增强"的理论背景。
- **vs DINOv2 + register（Darcet 等）**：他们用 register token 给特征图去噪但缺乏量化评估；本文的 Binary-AUC 第一次在 ImageNet 规模上把"register 是否真的让特征图更干净"量化出来，并发现大模型离群点反而拉低 AUC 的反直觉现象。

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一 $Y=MX$ + 秩层级 + Binary-AUC 的组合视角新颖，单点都不算首创但串成一条完整论证有价值。
- 实验充分度: ⭐⭐⭐⭐ 分类/分割/检测/鲁棒性 + 多粒度特征图分析覆盖较全，但限于 ImageNet 规模与单一深度配置。
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰、图示直观，公式排版偶有 OCR 噪声但逻辑链完整。
- 价值: ⭐⭐⭐⭐ 给"Vision Mamba 为什么work"一个可量化的代数解释，Binary-AUC 这一诊断工具有较强可复用性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](../../CVPR2025/others/care_transformer_linear_attention.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](../../ACL2025/others/the_hidden_attention_of_mamba_models.md)
- [\[CVPR 2026\] Modeling the Visual Ambiguity of Human Sketches](modeling_the_visual_ambiguity_of_human_sketches.md)
- [\[CVPR 2026\] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)
- [\[CVPR 2026\] Language Does Matter for Cross-Domain Few-Shot Visual Feature Enhancement](language_does_matter_for_cross-domain_few-shot_visual_feature_enhancement.md)

</div>

<!-- RELATED:END -->
