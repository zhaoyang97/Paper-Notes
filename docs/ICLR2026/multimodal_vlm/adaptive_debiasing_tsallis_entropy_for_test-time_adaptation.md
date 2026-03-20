# Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation

**会议**: ICLR 2026  
**arXiv**: [2602.11743](https://arxiv.org/abs/2602.11743)  
**代码**: [https://github.com/Jinx630/ADTE](https://github.com/Jinx630/ADTE)  
**领域**: 多模态VLM  
**关键词**: test-time adaptation, Tsallis entropy, CLIP, debiasing, uncertainty estimation  

## 一句话总结
提出将 Tsallis 熵（SE 的广义形式）引入 VLM 的 Test-Time Adaptation，并进一步发展为自适应去偏 Tsallis 熵（ADTE），为每个类别定制去偏参数 $q^l$，在不引入分布特定超参数的情况下比 Shannon 熵选择更可靠的高置信视图，在 ImageNet 及其 5 个变体和 10 个跨域 benchmark 上均超越 SOTA。

## 研究背景与动机
1. **领域现状**：TTA（Test-Time Adaptation）方法通过选择高置信增强视图来提升 CLIP 等 VLM 在分布外数据上的表现。代表方法如 TPT、Zero 等都使用 Shannon 熵来度量不确定性并筛选低熵视图。
2. **现有痛点**：CLIP 在不平衡的网络爬取数据上预训练，导致对头部类别过度自信、对尾部类别自信度不足。Shannon 熵对所有类别使用统一公式 $-p\log p$，无法区分不同类别的偏差程度，导致熵估计本身就是有偏的，进而影响高置信视图的选择质量。
3. **核心矛盾**：SE 假设概率分布是无偏的（广延性假设），但 CLIP 的预测分布存在系统性偏差（非广延性），SE 无法刻画这种偏差结构。
4. **本文要解决什么？** 如何在 TTA 过程中纠正 VLM 预测偏差对熵估计的影响？
5. **切入角度**：Tsallis 熵是 Shannon 熵的推广，通过非广延参数 $q$ 可以刻画概率分布间的统计依赖性。当 $q<1$ 时，TE 倾向于选择更可靠的高置信视图。
6. **核心idea一句话**：用 Tsallis 熵替代 Shannon 熵做高置信视图选择，并为每个类别自适应计算去偏参数 $q^l$。

## 方法详解

### 整体框架
ADTE 是 Zero/TPT 等 TTA 方法中 Shannon 熵的即插即用替代品。流程：测试图像 → N 个增强视图 → 用 ADTE 计算每个视图的不确定性 → 选择低熵的高置信视图 → 聚合预测。关键区别在于熵的计算方式和类别特定参数。

### 关键设计

1. **Tsallis 熵替代 Shannon 熵**:
   - 做什么：用 TE $\mathbf{H}_{TE} = \frac{\sum_l P_l^q - 1}{1-q}$ 替代 SE $\mathbf{H}_{SE} = -\sum_l P_l \log P_l$
   - 核心思路：理论证明当 $q \to 1$ 时 TE 退化为 SE（下界性质）；当 $q < 1$ 时，TE 选择的高置信视图有更高的 Top-K 累积可靠性（TcrK）；当 $0 < q < 1$ 时，TE 能自然缓解 VLM 偏差的影响
   - 设计动机：SE 对尾部类别（概率接近 0）的偏差敏感，TE 通过 $p^q$ 替代 $p\log p$ 改变了对小概率的处理方式

2. **Adaptive Debiasing Tsallis Entropy (ADTE)**:
   - 做什么：为每个类别 $l$ 定制特定参数 $q^l$，无需手动调优
   - 核心思路：(1) 通过维护 memory bank 估计类别先验概率 $\tilde{p}_l$（Jacobi 迭代求解，用伪标签近似）；(2) 将估计的偏差通过 min-max 归一化映射到 $[\alpha, \beta] = [0.01, 0.9]$ 区间作为 $q^l$——偏差越大的类别 $q^l$ 越小，纠正力度越大
   - 设计动机：手动调 $q$ 对不同测试分布不可行，且不同类别受偏差影响程度不同（头部 vs 尾部）

3. **与 Logit Adjustment 集成**:
   - ADTE 可与 logit adjustment 策略无缝结合：先用估计的偏差调整 logits，再用 ADTE 选择高置信视图
   - 整个过程不需要额外训练或分布特定的超参数调优

### 损失函数 / 训练策略
无需训练。ADTE 是纯推理时方法，直接替换 TTA pipeline 中的 Shannon 熵即可。Memory bank 大小为每类 10 个样本。

## 实验关键数据

### 主实验（ImageNet + 5 变体，CLIP ViT-B/16）

| 方法 | IN | IN-A | IN-R | IN-K | Average | OOD Avg |
|------|-----|------|------|------|---------|---------|
| CLIP | 68.7 | 50.6 | 77.7 | 48.3 | 61.5 | 59.7 |
| Zero | 70.9 | 64.0 | 80.8 | 50.3 | 66.2 | 65.0 |
| BCA | 70.2 | 61.1 | 80.7 | 50.9 | 65.6 | 64.4 |
| **ADTE** | **71.8** | **65.5** | **81.4** | **53.5** | **67.5** | **66.5** |

### 跨域 benchmark（10 个数据集最高平均性能）

| 指标 | 说明 |
|------|------|
| ADTE 平均准确率 | 10 个跨域 benchmark 上最高平均表现 |
| 模型无关 | 在 ViT-B/16 和 ViT-L/14 上都优于 SOTA |
| Prompt 无关 | 使用手工模板或 CuPL 生成的文本都有效 |

### 关键发现
- TE 当 $q < 1$ 时始终优于 SE（SE 是 TE 在 $q=1$ 的特例），但最优 $q$ 因测试分布而异
- ADTE 通过自适应 $q^l$ 消除了手动调参的需求，在所有测试分布上表现稳健
- 在 ImageNet-K 上提升最大（48.3→53.5），这是分布偏移最严重的变体
- ADTE 可以直接替换任何基于 SE 的 TTA 方法中的熵计算，无需其他修改

## 亮点与洞察
- **Shannon 熵的有偏性被首次系统分析**：在 VLM TTA 中，SE 隐含假设的广延性不成立，这个问题被忽视已久
- **Tsallis 熵作为直接替代品**：理论优雅（SE 是下界）且实际有效，且是即插即用的——任何用 SE 的 TTA 方法都可以直接换成 TE/ADTE
- **自适应参数估计的设计**：利用已有的偏差估计方法（来自 Frolic）转化为 $q^l$，复用了现有工具

## 局限性 / 可改进方向
- Memory bank 大小固定为每类 10 个，在类别极多（如 ImageNet 1000 类）时可能不够
- 偏差估计依赖伪标签质量，早期样本的伪标签可能不准
- 归一化区间 $[\alpha, \beta] = [0.01, 0.9]$ 仍是手动设定的超参数
- 仅在分类任务上验证，检测/分割等密集预测任务未覆盖

## 相关工作与启发
- **vs Zero/TPT**: ADTE 是它们的直接升级——仅替换熵计算即可获得提升，无需改动其他组件
- **vs Frolic**: Frolic 使用 logit adjustment 做偏差校正，ADTE 从熵估计层面做校正，两者互补
- **vs 传统 Tsallis 熵在域适应中的应用**: 以往工作在源域适应中优化 TE 做伪标签，ADTE 首次将其应用到 online TTA 的视图选择

## 评分
- 新颖性: ⭐⭐⭐⭐ Tsallis 熵在 VLM TTA 中的应用是新颖的理论视角
- 实验充分度: ⭐⭐⭐⭐⭐ ImageNet+5变体、10跨域benchmark、两个模型、两种prompt
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但公式密集
- 价值: ⭐⭐⭐⭐ 即插即用的 SE 替代品，实用性强
