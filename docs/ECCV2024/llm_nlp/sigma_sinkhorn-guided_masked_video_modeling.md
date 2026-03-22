# SIGMA: Sinkhorn-Guided Masked Video Modeling

**会议**: ECCV 2024  
**arXiv**: [2407.15447](https://arxiv.org/abs/2407.15447)  
**代码**: [https://quva-lab.github.io/SIGMA](https://quva-lab.github.io/SIGMA)  
**领域**: Self-supervised Video Learning  
**关键词**: masked video modeling, self-supervised learning, optimal transport, Sinkhorn algorithm, video representation

## 一句话总结

本文提出 SIGMA，通过引入投影网络将 masked video modeling 的重建目标从像素级升级为可学习的深层特征聚类分配，利用 Sinkhorn 算法的最优传输实施高熵正则化避免坍缩，在 10 个数据集 3 个 benchmark 上全面超越 VideoMAE 等 SOTA 方法。

## 研究背景与动机

1. **领域现状**: 视频自监督学习正经历从 pretext task →contrast learning → masked modeling 的演进。VideoMAE 等 masked video modeling (MVM) 方法通过重建被掩码的时空管（space-time tubes）的像素值来预训练 ViT，展现了很好的可扩展性。
2. **现有痛点**: 像素级重建目标本质上是低级的——因为单个 patch 或时空管不代表独立的语义单元（不像 NLP 中的词/子词），因此模型倾向于学习低级纹理特征而非高级语义。实验证据：VideoMAE 在 frozen linear probing 下性能很弱（K400 仅 20.7%）。
3. **核心矛盾**: 如果将重建目标换成可学习的深层特征，投影网络和视频模型联合优化时会坍缩到 trivial solution（所有输入映射到同一特征向量），L2 loss 无法防止这一点。
4. **切入角度**: 将特征空间约束为有限数量的可学习聚类中心，通过最优传输的等分约束（equipartition constraint）强制所有聚类被均匀使用，形成高熵瓶颈，既避免坍缩又为特征注入语义信息。
5. **核心 idea**: Sinkhorn 算法求解最优传输 → 生成伪标签 → 视频模型和投影网络对称预测彼此的聚类分配 → 联合学习语义丰富的特征空间。

## 方法详解

### 整体框架

输入视频 → 分割为 space-time tubes ($2 \times 16 \times 16$) → 90% 掩码 → 视频模型 $\Psi$（encoder-decoder）仅看未掩码部分，预测被掩码管的特征 $\mathbf{x}^\Psi$ → 投影网络 $\varphi$ 看到完整视频的所有管，输出特征 $\mathbf{x}^\varphi$ → 两组特征投影到共享的可学习 prototypes $\mathbf{C}$ → Sinkhorn 算法生成聚类伪标签 → 对称交叉熵损失。

### 关键设计

1. **从像素重建到特征重建**:
   - 做什么：用深层特征替代像素值作为 MVM 的重建目标
   - 核心思路：引入投影网络 $\varphi$ 嵌入所有 space-time tubes，选取被掩码管的特征作为目标。将 L2 重建损失改为在特征空间中：
     $$\mathcal{L}_2^F = \frac{1}{N}\sum_{i=1}^N \|\mathbf{x}_i^\varphi - \mathbf{x}_i^\Psi\|_2^2$$
   - 设计动机：像素级目标导致低级特征学习；深层特征目标可以捕获更抽象的语义信息
   - **问题**：联合优化两个网络会导致 trivial solution（所有空间时间管映射到同一特征向量）

2. **Sinkhorn-Guided Clustering Bottleneck**:
   - 做什么：将特征空间约束为有限聚类，防止坍缩
   - 核心思路：定义 $K$ 个可学习 prototypes $\mathbf{C} = \{\mathbf{c}_1, ..., \mathbf{c}_K\}$，将特征到 prototypes 的映射建模为基于熵正则化的最优传输问题：
     $$\min_{\mathbf{Q}} \langle \mathbf{Q}, -\log \mathbf{X} \rangle + \frac{1}{\lambda} \text{KL}(\mathbf{Q} \| rc^\top)$$
     其中等分约束 $r = \frac{1}{K} \cdot \mathbb{1}, c = \frac{1}{B} \cdot \mathbb{1}$ 强制所有 prototype 被均匀使用。使用快速 Sinkhorn-Knopp 算法在 GPU 上求解，生成软伪标签 $\mathbf{q}$。
   - 设计动机：(1) 有限数量的聚类中心迫使相似的时空管共享同一 prototype，注入语义信息；(2) 等分约束防止所有点坍缩到单一聚类；(3) 可以在 mini-batch 内在线计算，训练高效

3. **Symmetric Prediction Loss**:
   - 做什么：用伪标签构建自监督预测任务
   - 核心思路：视频模型预测投影网络的聚类分配，同时投影网络预测视频模型的聚类分配：
     $$\mathcal{L} = \frac{1}{B}\sum_{i=1}^B [\mathcal{L}_{CE}(\tilde{\mathbf{x}}_i^\varphi, \mathbf{q}_i^\Psi) + \mathcal{L}_{CE}(\tilde{\mathbf{x}}_i^\Psi, \mathbf{q}_i^\varphi)]$$
     其中 $\tilde{\mathbf{x}} = \mathbf{x}^\top \mathbf{c}$ 是特征与 prototype 的内积得分
   - 设计动机：(1) 对称预测避免了单向依赖；(2) 不需要动量编码器（EMA），简化了训练管线；(3) 不需要数据增强来生成不同视图

### 损失函数 / 训练策略

- 使用 AdamW 优化器，base lr $1.5e^{-4}$，weight decay 0.05，cosine decay
- 90% tube masking ratio，tube 大小 $2 \times 16 \times 16$
- 投影网络 $\varphi$ 可选：(a) 3 层 MLP（sigma-mlp），(b) 冻结的 DINO 预训练模型（sigma-dino）
- 预训练 800 epochs on K400/SSv2
- 温度参数 $\tau$ 控制 softmax 锐度

## 实验关键数据

### 主实验 — Frozen Linear Probing (ViT-B, K400 pretrained)

| 数据集 | SIGMA-MLP | SIGMA-DINO | VideoMAE | MGMAE | 提升 (MLP vs MAE) |
|--------|-----------|------------|----------|-------|----------|
| SSv2 | 19.9 | **20.8** | 17.5 | 16.8 | +2.4 |
| K400 | 30.7 | **47.5** | 20.7 | 24.9 | +10.0 |
| UCF-101 | 73.8 | **80.7** | 58.6 | 64.4 | +15.2 |
| HMDB-51 | 45.0 | **52.3** | 37.7 | 41.3 | +7.3 |
| IN-1K | 24.1 | **45.0** | 20.2 | 20.4 | +3.9 |
| C-100 | 46.2 | **66.7** | 40.4 | 44.1 | +5.8 |

### Full Finetuning (ViT-B, 800 epochs)

| 数据集 | SIGMA-DINO | VideoMAE | MME | MGMAE | 备注 |
|--------|------------|----------|-----|-------|------|
| SSv2 (K400 pretrain) | **71.1** | 68.5 | 70.5 | - | 超越所有包括用运动引导的方法 |
| SSv2 (SSv2 pretrain) | **70.9** | 69.6 | 70.0 | 71.0 | 与 MGMAE 持平（MGMAE 用了光流引导） |
| K400 | **81.6** | 80.0 | - | 81.2 | 不依赖运动引导即达 SOTA |

### 消融实验

| 配置 | Accuracy (mini-SSv2) | 说明 |
|------|---------|------|
| VideoMAE (baseline) | 56.9 | 像素级重建 |
| SIGMA-MLP + L2 loss | 36.4 | 深层特征 + L2 → 坍缩 |
| SIGMA-MLP + Eq.7 | **60.3** | Sinkhorn 聚类损失解决坍缩 |
| Prototypes K=1000 | 59.7 | 聚类太少 |
| Prototypes K=4000 | **60.3** | 最佳聚类数 |
| Prototypes K=6000 | 60.1 | 略降 |

### 关键发现

- **Frozen evaluation 差距巨大**: SIGMA-DINO 在 K400 frozen probing 上达 47.5%，VideoMAE 仅 20.7%，提升 130%——说明 SIGMA 学到的特征语义性远强于像素重建方法
- **L2 特征重建直接坍缩**: 在 mini-SSv2 上仅 36.4%（低于 VideoMAE 的 56.9%），彻底验证了 trivial solution 问题的严重性
- **无需运动引导**: SIGMA 不使用光流或运动向量，仅靠随机管掩码和 Sinkhorn 聚类就能匹配甚至超越使用运动引导的方法（MME、MGMAE）
- **无监督视频分割**: SIGMA-DINO 在 DAVIS 上 overclustering mIoU 达 56.5%（vs VideoMAE 50.9%），说明学到的特征具有更好的时空语义理解
- **SEVERE 泛化基准**: 在域偏移、样本效率、动作粒度、任务偏移四个维度上全面领先，平均分 64.3 vs VideoMAE 57.4

## 亮点与洞察

- **最优传输作为自监督正则器的思路极其优雅**: Sinkhorn 等分约束既防止feature collapse 又注入语义信息，一石二鸟
- **投影网络的灵活性**: 可以是简单 MLP（纯自监督）也可以是预训练 DINO（利用图像先验），框架统一
- **不依赖数据增强**: 不像 DINO、BYOL 等对比学习方法需要精心设计时空增强管线，SIGMA 通过掩码+聚类就够了，更易扩展
- **Frozen probing 是检验预训练特征质量的黄金标准**: 本文在这一指标上的优势远大于 full finetuning，说明特征本身的语义质量显著提升
- **时空管级别的聚类赋予了 token-level 的语义意义**：弥补了视觉 token（patch）与语言 token（word）之间的语义差距

## 局限性 / 可改进方向

- **未扩展到 ViT-L/ViT-H**: 受学术计算资源限制，仅验证了 ViT-S 和 ViT-B
- **DINO 变体引入了外部监督**: SIGMA-DINO 使用了 ImageNet 预训练的 DINO 模型，不是纯粹的视频自监督
- **聚类数 K 需要调参**: 虽然对 K 不太敏感，但最优值（~4000）仍需通过实验确定
- **SIGMA-MLP 在 full finetuning 下与 SIGMA-DINO 差距仍大**: 说明 MLP 投影网络的表达能力有限
- **可探索多层次聚类**: 当前仅用单层 prototype，层次化聚类可能捕获更丰富的语义结构

## 相关工作与启发

- **vs VideoMAE**: 核心区别在于重建目标——像素 vs 聚类伪标签。SIGMA 在所有评估协议上全面超越，尤其 frozen probing 差距巨大
- **vs MME**: MME 预测 HOG 特征 + 运动轨迹，需要预处理去除相机运动；SIGMA 不需任何预处理
- **vs MGMAE**: MGMAE 用运动向量指导掩码策略，是互补的技巧；SIGMA 也可结合运动指导掩码进一步提升
- **vs SwAV/DINO (图像域)**: SIGMA 将 Sinkhorn 聚类思路从图像对比学习成功迁移到视频 masked modeling
- **vs JEPA**: JEPA 也探索潜空间预测目标，但面临训练不稳定问题；SIGMA 通过 Sinkhorn 等分约束保持训练稳定

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Sinkhorn 聚类 + masked video modeling 的组合是首创，解决了深层目标学习的坍缩难题
- 实验充分度: ⭐⭐⭐⭐⭐ 10 个数据集、3 个 benchmark（含 frozen/finetuning/分割/泛化），极为充分
- 写作质量: ⭐⭐⭐⭐⭐ 从像素目标 → 特征目标 → 坍缩问题 → Sinkhorn 解决方案的逻辑链条非常自然
- 价值: ⭐⭐⭐⭐⭐ 对视频自监督学习领域有范式级影响，Sinkhorn 聚类思路可广泛迁移
