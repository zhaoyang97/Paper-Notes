# Accurate and Efficient Low-Rank Model Merging in Core Space

**会议**: NeurIPS 2025  
**arXiv**: [2509.17786](https://arxiv.org/abs/2509.17786)  
**代码**: [GitHub](https://github.com/apanariello4/core-space-merging)  
**领域**: 模型压缩 / LLM效率  
**关键词**: 模型合并, LoRA, 低秩映射, 参数高效微调, Core Space  

## 一句话总结
提出 Core Space Merging 框架——通过在低秩 LoRA 矩阵的公共参考基空间中进行模型合并，**无信息损失**地将合并操作从 $m \times n$ 全尺寸空间压缩到 $Tr \times Tr$ 紧凑空间（$T$ 为任务数，$r$ 为 LoRA 秩），在 Llama 3 8B 上达到 SOTA 合并精度同时计算成本降低数个数量级。

## 研究背景与动机

1. **领域现状**：随着模型规模增长，LoRA 等参数高效微调 (PEFT) 成为主流。模型合并旨在将多个任务专用 LoRA 合并为一个多任务模型，无需额外训练。
2. **现有痛点**：(a) 直接对 LoRA 矩阵做 Task Arithmetic 效果差——不同任务的 LoRA 基不对齐；(b) KnOTS 方法虽然在对齐空间合并，但需要对全尺寸矩阵做 SVD，计算复杂度为 $O(n^3 T^2)$，对大模型不可承受；(c) 用先进合并方法（TSV、Iso-C）在全空间操作同样极其昂贵。
3. **核心矛盾**：高效的合并（直接全空间 TA）精度差，高精度的合并（KnOTS + TSV）又丧失了 LoRA 的低秩效率优势。
4. **本文要解决什么？** 在保持低秩计算效率的同时实现高精度 LoRA 模型合并。
5. **切入角度**：观察到所有任务的 LoRA 更新共享一个公共子空间——找到这个子空间的参考基，在其中合并。
6. **核心 idea 一句话**：在 LoRA 低秩矩阵的 SVD 构成的公共参考基空间（Core Space）中合并，维度仅为 $Tr \times Tr$，严格无信息损失。

## 方法详解

### 整体框架
给定 $T$ 个任务的 LoRA 更新 $\{(A^{(t)}, B^{(t)})\}_{t=1}^T$ → 对堆叠的 $A^{(t)}$ 和 $B^{(t)}$ 做 SVD 得参考基 $(U_B^{ref}, V_A^{ref})$ → 将每个任务投影到 Core Space 得核矩阵 $M^{(t)} \in \mathbb{R}^{Tr \times Tr}$ → 在 Core Space 中用任意合并方法 $\mathcal{M}$ 合并 → 投影回原空间。

### 关键设计

1. **参考基构建 (Reference Bases)**:
   - 做什么：找到一组正交基，能同时无损表示所有任务的 LoRA 方向
   - 核心思路：将所有 $B^{(t)}$ 水平拼接、所有 $A^{(t)}$ 竖直拼接，分别做 SVD，取 $U_B^{ref} \in \mathbb{R}^{m \times Tr}$ 和 $V_A^{ref} \in \mathbb{R}^{n \times Tr}$
   - 设计动机：拼接后的 SVD 自然跨越所有任务子空间的联合空间
   - 关键优势：SVD 在 $\mathbb{R}^{Tr \times n}$ 而非 $\mathbb{R}^{Tn \times n}$（KnOTS）上操作，计算量大幅减少

2. **Core 矩阵映射 (Core Matrix Projection)**:
   - 做什么：将每个任务更新投影到紧凑的 $Tr \times Tr$ 空间
   - 核心思路：$M^{(t)} = (U_B^{ref\top} B^{(t)})(A^{(t)} V_A^{ref}) \in \mathbb{R}^{Tr \times Tr}$
   - 理论保证：通过最小二乘求解证明投影误差**严格为零**——$\|U_B^{ref} R_B^{(t)} - U_B^{(t)}\|_F^2 = 0$，因为参考基的列空间包含每个任务基的列空间
   - 重建公式：$\Delta W^{(t)} = U_B^{ref} M^{(t)} V_A^{ref\top}$，完全无损

3. **Core Space 中合并**:
   - 做什么：在 $Tr \times Tr$ 紧凑空间中执行任意合并方法
   - 核心思路：$M_{merged} = \mathcal{M}(\{M^{(t)}\}_{t=1}^T)$，然后 $\Delta W = U_B^{ref} M_{merged} V_A^{ref\top}$
   - 对线性合并方法（TA）：Core Space 合并和全空间合并**数学等价**
   - 对非线性方法（TIES、TSV、Iso-C）：Core Space 中操作**效果更好**，因为对齐后的表示减少了方向干扰

### 计算复杂度

| 方法 | TA | Iso-C | TSV |
|------|-----|-------|-----|
| Full Space | $O(n^2 Tr)$ | $O(n^3)$ | $O(n^3 T)$ |
| KnOTS | $O(n^3 T^2)$ | $O(n^3 T^2)$ | $O(n^3 T^2)$ |
| **Core Space** | $O(n^2 Tr)$ | $O(n^2 Tr + T^3 r^3)$ | $O(n^2 Tr + T^4 r^3)$ |

## 实验关键数据

### 主实验 (Llama 3 8B, 6 NLI tasks)

| 方法 | 空间 | 平均归一化准确率 | 时间(s) |
|------|------|------------------|---------|
| TA | Full/Core | ~90% | 基线 |
| TIES | Full | ~91% | 中等 |
| TSV | Full | ~93% | 极高 |
| TIES | KnOTS | ~92% | 极高 |
| TIES | **Core** | **~94%** | **低** |
| TSV+Iso-C | **Core** | **SOTA** | **低** |

### 消融 (ViT-B/32, 8 vision tasks)

| 合并空间 | TIES 归一化准确率 | TSV 归一化准确率 |
|----------|------------------|------------------|
| Full Space | 基线 | 基线 |
| KnOTS | +提升但慢 | +提升但极慢 |
| **Core Space** | **最高** | **最高** |

### 关键发现
- **在 Core Space 中非线性合并效果更好**：对 TIES、TSV 等非线性方法，Core Space 不仅快而且精度也更高
- **效率提升数个数量级**：在 Llama 3 8B 上，TSV 在 Core Space 中几秒完成，在 Full Space 需数小时
- **Core Space 维度极小**：对 $T=6, r=16$，Core 矩阵仅 $96 \times 96$（vs 全空间 $4096 \times 4096$）
- **无信息损失的严格证明**：不是近似，是数学上精确的零投影误差

## 亮点与洞察
- **"小空间大效果"的理论美感**：从联合子空间的角度出发，发现 $Tr \times Tr$ 空间足以无损表示所有任务信息，且在此空间中合并效果反而更好（因为更好的对齐）
- **与 KnOTS 的本质区别**：KnOTS 在全尺寸矩阵上做 SVD（$O(n^3 T^2)$），Core Space 在小矩阵上做 SVD（$O(n^2 Tr)$），效率差一个 $n/r$ 量级
- **对非线性方法的提升**：在对齐空间中操作，不同任务的方向干扰更小，TIES 的 sign conflict 减少
- **可迁移性**：与任何合并方法正交地组合，即插即用

## 局限性 / 可改进方向
- **假设 $Tr \leq \min(m,n)$**：任务数 × 秩不能超过模型维度，极多任务时可能需要截断
- **仅限 LoRA 适配**：对全量微调模型需要先做低秩近似
- **线性方法无提升**：对 TA 这种线性合并，Core Space 和 Full Space 数学等价，优势仅在非线性方法
- **改进方向**：(1) 自适应秩选择；(2) 扩展到其他 PEFT 方法（如 Adapter、Prefix Tuning）；(3) 研究 Core Space 中的最优合并策略

## 相关工作与启发
- **vs KnOTS**：同样在对齐空间合并，但 KnOTS 从全尺寸矩阵拼接做 SVD，Core Space 从 LoRA 低秩矩阵拼接做 SVD——本质差别在于利用了低秩结构
- **vs Task Arithmetic**：TA 是 Core Space 中线性合并的特例，但非线性方法在 Core Space 中有额外提升
- **vs TSV / Iso-C**：这些高级合并方法在 Core Space 中以极低成本运行，使其首次可扩展到大模型

## 评分
- 新颖性: ⭐⭐⭐⭐ Core Space 的概念优雅，无信息损失的证明是理论亮点
- 实验充分度: ⭐⭐⭐⭐ Vision (ViT-B/32, ViT-L/14) + Language (Llama 3 8B)，多种合并方法对比
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰严谨，无信息损失证明完整
- 价值: ⭐⭐⭐⭐⭐ 使高级模型合并方法首次可扩展到大模型，兼具理论和实用价值
