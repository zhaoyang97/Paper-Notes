# Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion

**会议**: NeurIPS 2025
**arXiv**: [2502.20120](https://arxiv.org/abs/2502.20120)
**代码**: [https://github.com/njustkmg/NeurIPS25-AUG](https://github.com/njustkmg/NeurIPS25-AUG)
**作者**: Qing-Yuan Jiang, Longfei Huang, Yang Yang（南京理工大学 / 南京大学）
**领域**: 多模态学习 · 模态不平衡
**关键词**: modality imbalance, classification ability disproportion, sustained boosting, adaptive classifier assignment

## 一句话总结

提出"**分类能力不均衡**"视角理解多模态学习中的模态不平衡，设计 Sustained Boosting 算法（共享编码器 + 多可配置分类器，同时优化分类和残差误差）配合自适应分类器分配（ACA），理论证明跨模态 gap loss 以 $\mathcal{O}(1/T)$ 收敛，在 CREMAD 等 6 个数据集上大幅超越 SOTA。

## 背景与动机

多模态学习（MML）的核心瓶颈是**模态不平衡**：联合训练时不同模态收敛速度差异显著。在 CREMAD 数据集上，音频（强模态）单模态准确率 ~63%，视频（弱模态）仅 ~45%，差距悬殊。现有解决方案分两类：

1. **调节学习过程**——OGM 做梯度调制、MSLR 调学习率、G-Blend 自适应融合权重，本质是放慢强模态 / 加速弱模态
2. **增强模态交互**——MLA 交替训练传递优化信息、ReconBoost 用梯度 boosting 捕获跨模态互补信息、DI-MML 注入跨模态优化信号

**关键洞察**：上述方法都在"平衡学习速度"层面做文章，忽视了更根本的问题——弱模态分类器的**分类能力本身就不足**。即使学习速度平衡了，弱分类器容量不够仍无法匹配强模态。

作者用 toy experiment 验证：对 naive MML 训练后的视频模态额外施加 gradient boosting（音频不动），视频准确率从 ~45% 跳到 ~65%+，整体准确率从 0.6507 升至远超 G-Blend 的水平。这证明**直接增强弱模态分类能力**是可行且有效的。

## 核心问题

如何在多模态联合训练框架中**直接提升弱模态的分类能力**，使强弱模态的分类性能趋于均衡，而非仅仅平衡学习速度？

## 方法

### 整体架构

每个模态使用：**共享编码器** $\phi^o(\cdot)$ 提取特征 $\boldsymbol{u}^o$ + **多个可配置分类器** $\psi_t^o(\cdot)$ 输出预测。编码器跨分类器共享参数，分类器最后一层（Layer2）跨模态共享以增强交互。

### 1. Sustained Boosting 算法

受 gradient boosting 启发，为弱模态训练 $n$ 个分类器，逐步学习前序分类器的残差。第 $t$ 个分类器学习的**残差标签**为：

$$\hat{\boldsymbol{y}}_{it}^o = \boldsymbol{y}_i - \lambda \sum_{j=1}^{t-1} \boldsymbol{y}_i \odot \boldsymbol{p}_{ij}^o$$

其中 $\lambda \in [0,1]$ 控制标签平滑程度，$\odot$ 为逐元素乘法，用 $\boldsymbol{y}_i$ 掩码确保残差非负。

总损失由三项组成：
- **残差误差** $\epsilon$：第 $t$ 个分类器对残差标签的交叉熵——学新信息
- **总体误差** $\epsilon_{\text{all}}$：所有 $t$ 个分类器预测之和对真实标签的交叉熵——保证整体准确
- **维护误差** $\epsilon_{\text{pre}}$：前 $t\!-\!1$ 个分类器之和对真实标签的交叉熵——防止共享编码器更新导致已有分类器退化

$$L(\boldsymbol{x}_i^o, \boldsymbol{y}_i, t) = \epsilon + \epsilon_{\text{all}} + \epsilon_{\text{pre}}$$

**与传统 gradient boosting 的核心区别**：传统方法是 stage-wise（逐阶段冻结），本文**持续同时优化**所有分类器与编码器，因此称为 "sustained" boosting。

### 2. 自适应分类器分配（ACA）

训练过程中模态间差距动态变化，固定分配分类器数量不够灵活。ACA 策略每 $t_N$ 个 epoch 用 confident score 检测：

$$s_t^o = \frac{1}{N}\sum_{i=1}^{N} \boldsymbol{y}_i^\top \left[\sum_{j=1}^{n^o} \boldsymbol{p}_{ij}^o\right]$$

若 $s_t^a - \sigma \cdot s_t^v > \tau$（音频远强于视频），给视频**新增一个分类器**；反之亦然。默认 $\sigma=1.0$，$\tau=0.01$。

### 3. 可配置分类器结构

每个分类器为轻量两层全连接：Layer1($D \times 256$) → ReLU → Layer2($256 \times K$)。新增一个分类器仅增加约 1M 参数（编码器 ResNet18 为 11.8M），代价极小。

### 4. 理论保证

定义跨模态 gap 函数 $\mathcal{G}(\Phi) = \mathcal{L}^a(\Phi^a) - \mathcal{L}^v(\Phi^v)$。在 Lipschitz 平滑等标准假设下：

$$\mathcal{G}(\Phi(T)) \leq \frac{\mathcal{G}(\Phi(0))}{1 + \frac{\nu^2\kappa^2}{2L_a\beta^2} T \cdot \mathcal{G}(\Phi(0))}$$

即 gap loss 以 $\mathcal{O}(1/T)$ 收敛——弱模态的损失确实会逐步追上强模态。

## 实验设计

**数据集**：6 个多模态数据集覆盖音视频（CREMAD、KSounds、VGGSound）、三模态手势（NVGesture: RGB+OF+Depth）、图文（Twitter、Sarcasm）。

**Baselines**：传统融合（Concat、Affine、ML-LSTM、Sum、Weight）+ 再平衡方法（MSES、G-Blend、MSLR、OGM、PMR、AGM、MMPareto、SMV、MLA、DI-MML、LFM、ReconBoost），共 17 个对比方法。

**实现细节**：
- 编码器：ResNet18（音视频）、I3D（NVGesture）、BERT + ResNet50（图文）
- 优化器：SGD，lr=0.01，momentum=0.9，weight decay=1e-4；图文用 Adam，lr=2e-5
- $\lambda$ 从 {0.1, 0.2, 0.33, 0.5, 1.0} 搜索
- $t_N$（检查间隔）：CREMAD 20 epochs，VGGSound/KSounds/NVGesture 10，Twitter 5，Sarcasm 1

## 实验结果

### 主实验

| 数据集 | 模态 | Naive MML | 最佳 baseline | **Ours** |
|--------|------|-----------|--------------|---------|
| CREMAD | 音频+视频 | 0.6507 | 0.8362 (LFM) | **0.8515** |
| KSounds | 音频+视频 | 0.6455 | 0.7253 (LFM) | **0.7263** |
| VGGSound | 音频+视频 | 0.5116 | 0.5274 (LFM) | **0.5301** |
| Twitter | 图+文 | 0.7300 | 0.7501 (LFM) | **0.7512** |
| Sarcasm | 图+文 | 0.8294 | 0.8497 (LFM) | **0.8510** |
| NVGesture | RGB+OF+Depth | 0.8237 | 0.8436 (LFM) | **0.8501** |

所有数据集上均取得最佳结果。CREMAD 上相比 Naive MML 提升 **20 个百分点**，相比此前最佳 LFM 仍高 1.5%。

### 消融实验（CREMAD）

| 损失组合 | 多模态 Acc | 音频 Acc | 视频 Acc |
|---------|-----------|---------|---------|
| 仅 $\epsilon$（残差） | 0.8333 | 0.6465 | 0.6734 |
| 仅 $\epsilon_{\text{all}}$（总体） | 0.8320 | 0.6573 | 0.6707 |
| 仅 $\epsilon_{\text{pre}}$（维护） | 0.8360 | 0.6841 | 0.6371 |
| **三者联合** | **0.8515** | 0.6835 | **0.6828** |

三项损失缺一不可，联合优化同时获得最高多模态精度与最均衡的模态表现。

### 自适应 vs 固定分类器

| 策略 | 音频/视频分类器数 | 多模态 Acc |
|------|----------------|-----------|
| 固定 10 个视频分类器 | 1+10 | 0.8091 |
| 固定 12 个视频分类器 | 1+12 | 0.8118 |
| **自适应（ACA）** | 1+10（动态） | **0.8515** |

相同最终分类器数下，自适应比固定策略高 **4.2%**，说明"何时加"比"加多少"更重要。

### 模型容量控制

| 方法 | 架构 | 参数量 | Acc |
|------|------|-------|-----|
| Naive MML | R18+R18 | 23.6M | 0.6507 |
| Naive MML | R18+R34 | 35.1M | 0.6277 |
| **Ours** | R18+R18+classifiers | **24.6M** | **0.8515** |

R18→R34 增加 11.5M 参数准确率反而下降（更难收敛）。本文仅增加 1M 参数即提升 20%——性能增益显然不来自参数量，而来自 boosting 机制。

### 模态缺失鲁棒性（CREMAD）

| 方法 | 缺失 0% | 缺失 20% | 缺失 50% |
|------|--------|---------|---------|
| Naive MML | 0.6507 | 0.5849 | 0.5242 |
| MLA | 0.7943 | 0.6935 | 0.5753 |
| **Ours** | **0.8515** | **0.7540** | **0.6008** |

50% 模态缺失下，本文方法（0.6008）仍优于 MLA 完整数据的结果（0.5753 @ 50% missing）。t-SNE 可视化也表明本文方法在视频模态上学到的特征区分性显著优于 naive MML 和 ReconBoost。

## 亮点

- **视角新颖**：从"分类能力不均衡"而非"学习速度不均衡"切入——抓住了模态不平衡的更本质原因
- **Boosting 迁移优雅**：将 gradient boosting 的残差学习无缝嵌入联合训练的多模态框架，通过 sustained 同步优化避免 stage-wise 的信息损失
- **理论闭环**：证明 gap loss 的 $\mathcal{O}(1/T)$ 收敛，为方法有效性提供理论保障
- **实验设计扎实**：模型容量控制实验排除了"参数更多"的替代解释，自适应 vs 固定对比说明动态分配的必要性

## 局限性

- 理论仅分析 boosting 对 gap 收敛的影响，完整框架的整体收敛性未证明
- 分类器数量随训练递增，推理时存在线性开销
- 仅验证分类任务，检索/生成/分割等下游任务待探索
- 对 early fusion 架构的适配方案未展开讨论
- 超参数 $t_N$ 按数据集手动设定，缺乏自适应选择机制

## 与相关工作的关键区别

- **vs OGM**：OGM 通过梯度调制放慢强模态学习；本文直接增强弱模态分类器容量——更根本
- **vs MLA**：MLA 交替训练传递优化信息弥合差距；本文通过 boosting 显式提升弱模态——CREMAD 上高 1.5% (vs LFM)
- **vs ReconBoost**：同样用 gradient boosting，但 ReconBoost 目标是迭代捕获跨模态互补信息；本文目标是直接提升弱模态分类能力——出发点和机制均不同
- **vs 增大网络容量**：R18→R34 增加 11.5M 参数反而更差；多分类器集成仅加 1M 参数效果远优——结构比容量重要

## 评分

- **新颖性**: ⭐⭐⭐⭐ 分类能力不均衡视角 + sustained boosting 在 MML 中的应用有明确新意
- **实验**: ⭐⭐⭐⭐⭐ 6 数据集 17 对比方法 + 消融 + 策略对比 + 容量控制 + 缺失鲁棒性 + t-SNE 可视化
- **写作**: ⭐⭐⭐⭐ toy experiment 直觉清晰，问题→方法→理论逻辑链完整
- **影响力**: ⭐⭐⭐⭐ 为多模态不平衡提供了新的理论框架和实用方法，CREMAD 上 20% 绝对提升令人瞩目
