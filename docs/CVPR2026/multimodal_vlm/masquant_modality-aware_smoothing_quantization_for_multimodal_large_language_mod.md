# MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models

**会议**: CVPR 2026
**arXiv**: [2603.04800](https://arxiv.org/abs/2603.04800)
**代码**: https://github.com/alibaba/EfficientAI
**领域**: 多模态VLM
**关键词**: 后训练量化, 多模态LLM, 平滑量化, 跨模态补偿, 低秩分解

## 一句话总结
揭示了通道平滑量化（如 SmoothQuant）直接应用于 MLLM 时的"平滑失配"问题——不同模态激活幅度差异巨大导致非主导模态被过度平滑，提出 MASQuant 通过模态感知平滑因子和基于 SVD 白化的跨模态低秩补偿解决该问题。

## 研究背景与动机

1. **领域现状**：后训练量化（PTQ）是部署大模型的关键技术。基于计算不变性的通道平滑方法（SmoothQuant、AWQ 等）在纯文本 LLM 上表现优异，通过通道缩放因子重分配激活离群值。
2. **现有痛点**：直接将通道平滑应用于 MLLM 时，视觉 token 的激活幅度通常比文本 token 大 10-100 倍。统一的平滑因子由主导模态（通常是视觉）决定，导致非主导模态（文本、音频）被过度平滑，信号被压缩，产生严重量化误差——即"平滑失配"（Smoothing Misalignment）。
3. **核心矛盾**：为每个模态学习独立平滑因子解决了失配问题，但需要为每个模态存储独立的量化权重，完全违背了量化压缩的初衷。
4. **本文要解决什么**：能否在保持单一量化权重的前提下，实现模态感知的平滑量化？
5. **切入角度**：观察到不同模态平滑后的权重差异是低秩的（可数学证明），因此可用轻量低秩矩阵补偿。
6. **核心 idea**：学习模态特异平滑因子 + 以文本模态为基准存储一套量化权重 + 用 SVD 白化低秩补偿其他模态。

## 方法详解

### 整体框架
MASQuant 包括两个核心模块：(1) Modality-Aware Smoothing (MAS) 为每个模态学习独立优化的平滑因子；(2) Cross-Modal Compensation (CMC) 通过 SVD 白化将跨模态权重差异压缩为低秩形式，仅存储一套量化权重 + 轻量补偿矩阵。

### 关键设计

1. **Modality-Aware Smoothing (MAS)**：
   - 做什么：为每个模态 $m$ 学习独立的优化平滑因子 $\mathbf{S}_m$
   - 核心思路：初始化 $s_i^m = \sqrt{\max_t|x_{t,i}^m| / \max_j|w_{j,i}|}$，然后通过最小化模态特定的 MAE 损失 $\sum_{m} \lambda_m \cdot \mathcal{L}_{MAE}(\mathbf{S}_m, \mathbf{X}_m, \mathbf{W})$ 直接优化平滑因子
   - SQNR 理论分析：证明了统一平滑导致非主导模态 SQNR 退化 $\Delta = 10\log_{10}(\frac{d(\min_i \alpha_i^2)}{\sum_i 1/\alpha_i^2})$，其中 $\alpha_i$ 为模态间激活范围比
   - 设计动机：不再搜索超参 $\beta$，而是直接优化平滑因子本身，达到通道平滑的优化极限

2. **Cross-Modal Compensation (CMC)**：
   - 做什么：使用单一量化权重的同时补偿非文本模态的量化误差
   - 核心思路：以文本模态平滑权重 $Q(\mathbf{S}_t \mathbf{W})$ 为基准，视觉模态产生残差 $\Delta\mathbf{W} = \mathbf{S}_v \mathbf{W} - Q(\mathbf{S}_t \mathbf{W})$。直接对 $\Delta\mathbf{W}$ 做 SVD 效果差（缺乏低秩结构），但通过白化变换 $\mathbf{T} = (\mathbf{P}\Lambda^{1/2})^\top$ 后，$\mathbf{T}(\Delta\mathbf{W})$ 呈现强低秩特性
   - 截断 SVD 后得到 $\Delta\mathbf{W} \approx \mathbf{L}_1 \mathbf{L}_2$，其中 $\mathbf{L}_1 = \mathbf{T}^{-1}\mathbf{U}_r$，$\mathbf{L}_2 = \Sigma_r \mathbf{V}_r^\top$
   - 理论保证：证明了该方案最小化输出重建误差 $\|\mathbf{X}_v \mathbf{S}_v^{-1}(\Delta\mathbf{W} - \mathbf{L})\|_F^2$

3. **推理流程**：
   - 文本模态：$\mathbf{Y} = Q(\mathbf{X}_t \mathbf{S}_t^{-1}) \cdot Q(\mathbf{S}_t \mathbf{W})$
   - 非文本模态：$\mathbf{Y} = Q(\mathbf{X}_m \mathbf{S}_m^{-1}) \cdot Q(\mathbf{S}_t \mathbf{W}) + \mathbf{X}_m \mathbf{S}_m^{-1} \cdot \mathbf{L}_1^m \mathbf{L}_2^m$
   - 仅需额外存储轻量低秩矩阵，主权重仍是单一量化版本

## 实验关键数据

### 主实验（Qwen2.5-VL 系列）

| 方法 | Bits | MMMU | OCRBench | ScienceQA | TextVQA | Avg |
|------|------|------|----------|-----------|---------|-----|
| FP16 | W16A16 | 基线 | 基线 | 基线 | 基线 | 100% |
| SmoothQuant | W8A8 | 下降明显 | 下降 | 下降 | 下降 | - |
| MASQuant | W8A8 | **最优** | **最优** | **最优** | **最优** | SOTA |

### 跨架构验证

| 模型类型 | 说明 |
|---------|------|
| 双模态 VLM | Qwen2.5-VL-3B/7B 上一致优于 SmoothQuant、AWQ |
| 三模态 Omni | Qwen2.5-Omni-3B 上同样有效，音频模态也受益 |

### 消融实验
- MAS 单独使用即显著提升 SQNR（图 2 验证定理 1）
- CMC 的低秩近似质量随秩增加快速收敛
- 白化后残差的低秩特性远优于直接 SVD

## 亮点
- 首次形式化定义 MLLM 量化中的“平滑失配”问题并给出 SQNR 理论分析（定理 1）
- 数学证明跨模态激活差异的低秩特性，使 CMC 有理论保证（定理 2）
- 框架同时适用于双模态（视觉-文本）和三模态（视觉-文本-音频）MLLM
- 保持单一量化权重，额外存储开销极低（仅低秩矩阵）
- 在 Qwen2.5-VL 和 Qwen2.5-Omni 上均一致优于现有通道平滑 PTQ 方法

### 消融实验
- 仅用 MAS（不加 CMC）：不同模态需存储独立量化权重，但量化精度最优
- 仅用 CMC（不改平滑）：修补效果有限，因底层平滑失配未解决
- MAS + CMC（完整方案）：在单一权重约束下逼近 MAS 的精度上限
- CMC 低秩补偿：秩 16-32 通常足以恢复 90%+ 的精度差距
- 白化后 $\mathbf{T}(\Delta\mathbf{W})$ 的奇异值衰减远快于直接 SVD，验证了低秩假设

## 局限性 / 可改进方向
- 校准阶段需要收集各模态数据来分别优化平滑因子，增加预处理复杂度
- 低秩补偿的秩 $r$ 选择需要在精度和额外存储之间权衡
- 当前仅验证了 W8A8 和 W4A8 设置，更激进的低位宽（如 W2A4）效果未知
- 非文本模态推理时需要额外的矩阵乘法 $\mathbf{X}_m \mathbf{S}_m^{-1} \cdot \mathbf{L}_1^m \mathbf{L}_2^m$，有少量延迟开销
- 可考虑将模态感知思想推广到旋转基方法（如 QuaRot、SpinQuant）
- 三模态及以上场景中低秩补偿矩阵数量线性增长，需要内存管理优化

### 实现细节
- MAS 优化使用 Adam，通常 100-200 次迭代即可收敛
- CMC 低秩矩阵以 FP16 存储，相比完整权重矩阵占用可忽略不计

