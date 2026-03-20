# Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training

**会议**: NeurIPS 2025  
**arXiv**: [2505.23971](https://arxiv.org/abs/2505.23971)  
**代码**: 无  
**领域**: LLM效率 / 训练优化  
**关键词**: critical batch size, large-batch training, learning rate scaling, batch size warmup, gradient noise scale, OLMo

## 一句话总结

提出 branched training 方法直接实证测量临界 batch size (CBS)，发现 CBS 在训练早期快速增长后趋于平稳且不依赖模型规模，据此设计 batch size warmup 策略以 43% 更少的梯度步数达到同等甚至更优的训练 loss。

## 研究背景与动机

1. **领域现状**：大 batch 训练通过增加数据并行性来提升 LLM 训练吞吐量，是大规模预训练的核心需求。临界 batch size (CBS) 作为"不显著降低 token 效率的最大 batch size"是平衡效率与性能的关键概念。
2. **现有痛点**：McCandlish et al. (2018) 提出用梯度噪声尺度 (gradient noise scale) 作为 CBS 的代理指标，被 GPT-3 等工作采用。但该方法依赖两个强假设：(a) 使用 SGD 优化器；(b) 梯度是良条件的 (well-conditioned)。
3. **核心矛盾**：实际 LLM 训练使用 Adam 优化器，理论分析表明 Adam 应遵循平方根缩放规则而非线性缩放规则。此外，梯度噪声尺度的简化形式 $\mathcal{B}_{\text{simple}}$ 需要 Hessian 为单位矩阵的假设才能等价于真正的 CBS。
4. **本文要解决什么**：如何在不依赖强假设的前提下直接测量 CBS？CBS 在训练过程中如何变化？如何利用 CBS 信息制定实用的大 batch 训练策略？
5. **切入角度**：用 branched training 直接实证逼近 CBS，避免间接代理带来的不可靠性。
6. **核心idea**：CBS 是训练过程中的动态量，从近零增长到平台值，这一洞察自然引出 batch size warmup 策略。

## 方法详解

### 整体框架

从训练中的任意 checkpoint 分叉出多条训练分支，每条使用不同的 batch size（基础 BS 的 k 倍），训练固定的 Δ 个 token 后比较 loss，找到 loss 不显著增加的最大 batch size 即为当前 checkpoint 处的局部 CBS。

### 关键设计

1. **Branched Training 测量 CBS**
   - **做什么**：从 checkpoint 出发，以不同的 batch size 倍数 k 训练 Δ=2B tokens，比较最终 loss
   - **核心思路**：直接观察 loss 对 batch size 的响应，而非通过梯度统计量间接推断
   - **设计动机**：避免梯度噪声方法的两个强假设（SGD + 良条件梯度），只需一个弱假设——如果在 Δ token 内 loss 能恢复，则此 batch size 在长期训练中也可以

2. **局部恢复假设 (Local Recovery Assumption)**
   - **做什么**：假设如果某 batch size B 在 Δ token 的窗口内能恢复到与小 batch size 相近的 loss，那么它在更长训练中也不会退化
   - **核心思路**：将全局 CBS 测量简化为局部测量，只需少量额外训练即可
   - **设计动机**：避免像先前工作那样需要为每个 batch size 跑完整训练流程

3. **CBS 检测标准**
   - **做什么**：用 tolerance ε=0.01 判断 loss 是否显著上升，用指数移动平均 (α=0.5) 平滑 loss 噪声
   - **核心思路**：以 loss 对 k 的曲线为依据，找到 loss 开始上升的拐点 k*，对应的 B* = k* × B_base 即为 CBS
   - **设计动机**：预训练 loss 噪声较大，需要平滑处理；tolerance 防止因噪声导致误判

4. **Batch Size Warmup 策略**
   - **做什么**：训练初始 BS=1024，当测得 CBS 超过 2×当前 BS 时翻倍 BS，同时按平方根规则调整 LR
   - **核心思路**：CBS 在训练早期快速增长，跟踪 CBS 增长来逐步增大 BS，既享受大 batch 的吞吐量优势又不损性能
   - **设计动机**：CBS 从近零开始意味着训练开始时不能用大 batch，但 CBS 很快增长到数千，大部分训练时间可用大 batch

### 损失函数 / 训练策略

- 学习率缩放规则：使用 Adam 的平方根缩放 $\eta' = \sqrt{B'/B} \cdot \eta$（而非 SGD 的线性缩放）
- BS 翻倍时间节点：基于 CBS 测量，在 168B tokens 处翻倍至 2048，在 503B tokens 处翻倍至 4096
- 叠加在原始余弦学习率 schedule 之上，兼容现有训练流程
- Mid-training 阶段：从预训练最终 checkpoint 线性退火 50B tokens 至 LR=0

## 实验关键数据

### 主实验

| 方法 | 预训练 Loss ↓ | Mid-training Loss ↓ | 梯度步数节省 ↑ |
|------|-------------|---------------------|--------------|
| Batch Size Warmup (本文) | **2.5891** | **2.5433** | 43% |
| 固定小 Batch (BS=1024) | 2.6057 | 2.5486 | 0% |
| 固定大 Batch (BS=4096) | 2.5962 | 2.5506 | 75% |

### 消融实验 / OOD评估

| 方法 | Task BPB (PT/MT) ↓ | C4 Loss (PT/MT) ↓ | Pile Loss (PT/MT) ↓ |
|------|-------------------|-------------------|---------------------|
| BS Warmup | 1.0316 / **1.0076** | **2.8049** / 2.7597 | 2.1916 / 2.1521 |
| 小 Batch 对照 | **1.0112** / 0.9999 | 2.8196 / 2.7622 | 2.2073 / 2.1471 |
| 大 Batch 对照 | 1.0571 / 1.0193 | 2.8107 / 2.7658 | 2.1996 / 2.1586 |

### 关键发现

1. **CBS 动态特性**：CBS 在初始化时接近 0，在前 50k 步内快速增长，然后平稳于约 4096 documents（每 doc 4096 tokens）。1B 和 7B 模型呈现相同定性趋势。
2. **梯度噪声不可靠**：梯度噪声尺度在两个模型上都大幅低估 CBS（差数个数量级），且定性趋势不吻合（尤其 7B 模型），不应用作 CBS 代理。
3. **Warmup 有效性**：BS warmup 在预训练和 mid-training 后均达到略优于小 batch 对照的 loss，同时使用 43% 更少的梯度步数。固定大 batch 虽节省 75% 步数但 loss 退化。
4. **CBS 与模型规模无关**：1B 和 7B 的 CBS 曲线定性一致，支持先前工作关于 CBS 主要随数据量而非模型大小缩放的结论。
5. **理论推导**：若局部 CBS 按 $t^{1/2}$ 增长，可推导出全局固定 CBS 按 $\sqrt{T}$ 缩放的规律，与先前工作一致。

## 亮点与洞察

- **CBS 是动态量而非常数**：这是本文最核心的洞察。训练初期 CBS 很小，固定大 batch 会在开头阶段超过 CBS 导致 loss 退化；大部分训练过程中 CBS 又够大，可以安全使用大 batch。Warmup 精准利用了这一规律。
- **方法极其简单且可靠**：branched training 无需任何额外假设，只需少量分叉训练即可测量 CBS，比梯度噪声方法更可信。
- **Adam 的平方根缩放**：本文为 Adam 优化器下的 batch size-LR 关系提供了实证验证。这与 Malladi et al. (2022) 的理论分析一致，是对实践的重要修正（GPT-3 用了线性缩放）。
- **跨规模一致性意义重大**：1B 测得的 CBS 可直接指导 7B 甚至更大规模的训练配置，极大降低了测量成本。

## 局限性 / 可改进方向

1. **Branched training 的额外成本**：虽然比全训练便宜，但仍需为每个 checkpoint 跑多条分支。未来可以探索在线测量 CBS 的方法。
2. **窗口大小 Δ 的敏感性**：较大的 Δ 可能产生更大的 CBS 估计值，目前未系统分析 Δ 的影响。
3. **仅 2 的幂次翻倍**：BS 只能倍增导致与真实 CBS 的匹配较粗糙，更细粒度的调整可能进一步提升效率。
4. **规模覆盖有限**：仅验证了 1B 和 7B，70B+ 规模的行为待确认。
5. **仅 608B tokens 预训练**：完整 OLMo 训练使用 4T tokens，当前实验仅覆盖约 15%，更长训练的 CBS 行为待验证。
6. **翻倍阈值选择手动**：目前基于人工读图选择翻倍点，缺乏系统化的阈值设定方法论。

## 相关工作与启发

- **McCandlish et al. (2018)**：提出梯度噪声尺度估计 CBS，被 GPT-3 采用但本文证明不可靠
- **Zhang et al. (2024)**：CBS 主要随数据量缩放（$\propto \sqrt{D}$），本文从局部 CBS 角度提供了一致的解释
- **Malladi et al. (2022)**：从 SDE 角度论证 Adam 应使用平方根缩放规则，本文实践验证了这一理论
- **Smith et al. (2018)**：探索用增大 batch size 替代降低学习率，概念相关但本文确保 BS 不超过 CBS
- **启发**：该方法可以扩展到其他需要确定最优 batch size 的场景（如微调、RLHF）；CBS-aware 的调度器可以成为标准训练流水线的一部分

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 方法非常简单（分叉训练比较 loss），但洞察深刻（CBS 是动态量 + 梯度噪声不靠谱），简单方法解决了重要问题
- **实验充分度**: ⭐⭐⭐⭐ — 系统测量了多个 checkpoint × 多个 batch size 倍数 × 两个模型规模，还有 OOD 评估和 mid-training 验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 动机清晰、假设明确说明、图表信息量大，是方法论论文的典范
- **价值**: ⭐⭐⭐⭐ — 对 LLM 预训练实践有直接指导意义。Warmup 策略无侵入性，可与现有训练流程无缝集成
