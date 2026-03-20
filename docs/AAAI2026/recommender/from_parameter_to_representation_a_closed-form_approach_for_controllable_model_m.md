# From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging

**会议**: AAAI 2026
**arXiv**: [2511.10943](https://arxiv.org/abs/2511.10943)
**代码**: 无
**领域**: 对齐RLHF / 模型压缩
**关键词**: model merging, controllable merging, Pareto front, representation correction, multi-objective optimization

## 一句话总结
提出 ReACT，将可控模型合并从参数空间优化转移到表征空间校正，通过闭式解实现任意用户偏好下的 Pareto 最优模型即时生成，比现有方法快 36-208 倍且性能更优。

## 研究背景与动机

1. **领域现状**：模型合并（Model Merging）通过将多个任务专用模型合并为一个通用模型来避免昂贵的多任务重训练。现有方法包括权重平均、Task Arithmetic、TIES-Merging、AdaMerging 等。
2. **现有痛点**：传统方法产出静态的"one-size-fits-all"模型，无法让用户控制不同任务间的性能权衡。Pareto Merging (PM) 和 MAP 虽实现了可控合并，但依赖 compile-then-query 范式——PM 需要复杂的迭代训练，MAP 使用进化搜索且复杂度随任务数指数增长。
3. **核心矛盾**：可控合并方法在参数空间中进行昂贵的多目标优化，每次任务集变化都需从头重做，实际部署代价极高。
4. **本文要解决什么？** 如何以极低计算成本实现灵活的偏好感知模型生成？
5. **切入角度**：作者发现模型合并后的性能退化本质上源于表征空间的**全局线性畸变**（旋转+缩放），而非复杂的非线性变形。
6. **核心 idea 一句话**：在表征空间构建线性校正映射，推导闭式 Pareto 最优解，绕过一切迭代优化。

## 方法详解

### 整体框架

ReACT 分为两个阶段：
- **离线阶段**：给定预合并骨干 $f_{\text{merge}}$ 和 $T$ 个任务专家模型 $\{f_t\}$，对每个任务收集少量校准数据，提取合并模型表征 $\mathcal{Z}_t^{\text{mtl}}$ 和专家模型表征 $\mathcal{Z}_t^{\text{ind}}$，计算可复用的校正矩阵组件 $\{\hat{W}_t, C_t\}$。
- **在线阶段**：给定用户偏好向量 $\mathbf{p}$，通过闭式公式即时组装校正矩阵 $W_\mathbf{p}$，对推理时的表征做一次线性变换即可。

### 关键设计

1. **线性表征校正（Linear Representation Correction）**
   - 做什么：用线性变换矩阵 $W_t$ 将合并模型的表征映射回任务专家的表征空间
   - 核心思路：对每个任务 $t$，求解 $\min_{W_t} \|W_t \mathcal{Z}_t^{\text{mtl}} - \mathcal{Z}_t^{\text{ind}}\|_F^2$，即最小化校正后表征与专家表征之间的 Frobenius 范数
   - 设计动机：t-SNE 可视化显示合并模型表征与专家表征的偏差主要是全局旋转和缩放（线性畸变），不需要非线性方法

2. **最优正交正则化（Optimal Orthogonal Regularization）**
   - 做什么：用正交 Procrustes 解作为正则化先验，防止校正矩阵过拟合
   - 核心思路：正则化目标为 $\min_{W_t} \|W_t \mathcal{Z}_t^{\text{mtl}} - \mathcal{Z}_t^{\text{ind}}\|_F^2 + \beta\|W_t - W_t^{\text{orth}}\|_F^2$，其中 $W_t^{\text{orth}}$ 通过 SVD 的 Procrustes 解获得。闭式解为 $\hat{W}_t = (\mathcal{Z}_t^{\text{ind}} {\mathcal{Z}_t^{\text{mtl}}}^\top + \beta W_t^{\text{orth}})(\mathcal{Z}_t^{\text{mtl}} {\mathcal{Z}_t^{\text{mtl}}}^\top + \beta I)^{-1}$
   - 设计动机：校准数据通常很少（测试时无标签数据），纯最小二乘易过拟合。正交约束保持表征空间几何结构不被扭曲。

3. **Pareto 最优表征校正（Pareto-Optimal Representation Correction）**
   - 做什么：给定用户偏好 $\mathbf{p}$，生成全局最优的校正矩阵
   - 核心思路：通过线性标量化将多目标问题转化为单目标：$\min_W \sum_{t=1}^T p_t L_t(W)$。由于每个 $L_t(W)$ 都是凸二次函数，闭式解为 $W_\mathbf{p} = (\sum_t p_t \hat{W}_t C_t)(\sum_t p_t C_t)^{-1}$，其中 $C_t = \mathcal{Z}_t^{\text{mtl}} {\mathcal{Z}_t^{\text{mtl}}}^\top + \beta I$
   - 设计动机：不是简单的加权平均 $\sum p_t \hat{W}_t$，而是通过数据感知权重矩阵 $C_t$ 让特征结构更显著的任务获得更大影响力

### 损失函数 / 训练策略

无需训练，整个流程基于闭式计算：SVD + 矩阵求逆。超参数仅有正则化强度 $\beta$（默认 0.1），对值域鲁棒。复杂度为 $O(T D_{\text{rep}}^3)$，对 ViT-B/32 的 $D_{\text{rep}}=512$ 仅需毫秒级计算。

## 实验关键数据

### 主实验（8 个 ViT-B/32 模型合并）

| 偏好设置 | 方法 | Avg Acc (%) | 说明 |
|---------|------|------------|------|
| Equal | AMPP+PM | 85.2 | 现有 SOTA |
| Equal | AMPP+Ours | **85.4** | +0.2% |
| Priority | AMPP+PM | 85.9 | — |
| Priority | AMPP+Ours | **87.3** | +1.4% |
| One-hot | AMPP+PM | 83.6 | — |
| One-hot | AMPP+Ours | **88.9** | +5.3% |

### 消融实验

| 配置 | Equal NAcc | HV | 说明 |
|------|-----------|-----|------|
| Full model (Ours) | 93.5 | 70.2 | 完整模型 |
| Naive aggregation | 92.2 | 67.6 | 简单加权平均，缺少数据感知权重 |
| Polar decomposition | 91.5 | 64.0 | 分解旋转和缩放，效果更差 |
| $\beta=0$（无正则化） | ~82 | — | 过拟合校准数据 |
| 纯正交 ($\beta \to \infty$) | ~94 | — | 缺乏数据驱动的细粒度校正 |

### 关键发现
- **One-hot 偏好场景优势最大**（+5.3%），说明方法在极端偏好下不会像 PM 一样崩溃
- 仅用 **10% 无标签校准数据** 就超过全量数据的 PM 基线
- 数据感知聚合比 naive 加权平均高 2.6 HV，$C_t$ 权重矩阵是核心
- 计算效率：8 任务合并仅需 0.056 GPU 小时，PM 需 2.0 小时（36x），MAP 需 11.6 小时（208x）

## 亮点与洞察
- **闭式解替代迭代优化**：将可控模型合并从搜索/训练问题转化为矩阵代数问题，优雅的范式转变。核心洞察是表征畸变是全局线性的。
- **数据感知聚合**：$W_\mathbf{p}$ 不是 $\sum p_t \hat{W}_t$，而通过 $C_t$ 矩阵自然为特征方差大的任务赋予更高权重。
- **可迁移思路**：表征空间校正范式可迁移到持续学习中旧任务与新任务的表征对齐，或联邦学习中不同客户端模型的表征融合。

## 局限性 / 可改进方向
- **线性标量化的局限**：无法覆盖凹 Pareto 前沿的所有点
- **依赖无标签校准数据**：虽然仅需 10%，但仍需测试时的少量数据，完全 zero-shot 不可行
- **线性假设的适用性**：仅在 ViT-B/32 上验证，对更大模型或非 CLIP 架构是否成立未知
- **仅验证分类任务**：8 个数据集都是图像分类，对检测、分割、生成等未验证

## 相关工作与启发
- **vs Pareto Merging (PM)**: PM 在参数空间做迭代训练学习 Pareto 集低秩表示，本文在表征空间用闭式解，效率高 36 倍且 one-hot 性能高 5.3%
- **vs MAP**: MAP 用进化搜索，复杂度指数增长，本文线性增长
- **vs Representation Surgery**: RS 用 per-sample 的 MLP 做非线性校正，本文证明全局线性足够且更 data-efficient

## 评分
- 新颖性: ⭐⭐⭐⭐ 参数空间到表征空间的视角转换加闭式解是独特贡献
- 实验充分度: ⭐⭐⭐⭐ 8 数据集、多偏好设置、消融完整，但仅限分类和 ViT-B/32
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，可视化丰富，故事线流畅
- 价值: ⭐⭐⭐⭐ 提供轻量优雅的可控合并方案，适用范围待扩展
