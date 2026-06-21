---
title: >-
  [论文解读] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning
description: >-
  [ICLR 2026][PU学习] 提出 NcPU 非对比 PU 学习框架，通过对标准非对比损失做 sqrt 变换（NoiSNCL）让 clean pair 梯度主导训练、用 PhantomGate 提供保守负监督并支持 regret 回退，两个模块在 EM 框架下迭代互利；在不依赖辅助负样本或预估类先验的前提下，CIFAR-100 上与监督学习差距从 14.26% 缩至 <1.4%，xBD 灾损评估上同样达到 SOTA。
tags:
  - "ICLR 2026"
  - "PU学习"
  - "非对比表示学习"
  - "噪声对鲁棒"
  - "伪标签消歧"
  - "EM框架"
---

# Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning

**会议**: ICLR 2026  
**arXiv**: [2510.01278](https://arxiv.org/abs/2510.01278)  
**代码**: [https://github.com/Hengwei-Zhao96/NcPU](https://github.com/Hengwei-Zhao96/NcPU)  
**领域**: 其他 / 弱监督学习  
**关键词**: PU学习, 非对比表示学习, 噪声对鲁棒, 伪标签消歧, EM框架

## 一句话总结

提出 NcPU 非对比 PU 学习框架，通过对标准非对比损失做 sqrt 变换（NoiSNCL）让 clean pair 梯度主导训练、用 PhantomGate 提供保守负监督并支持 regret 回退，两个模块在 EM 框架下迭代互利；在不依赖辅助负样本或预估类先验的前提下，CIFAR-100 上与监督学习差距从 14.26% 缩至 <1.4%，xBD 灾损评估上同样达到 SOTA。

## 研究背景与动机

**领域现状**：PU 学习（Positive-Unlabeled Learning）只有少量已标注正样本和大量无标签数据，需要训练一个正/负二分类器。典型应用场景包括：灾后遥感建筑损伤识别中只有部分受损建筑被标注、产品推荐中只有点击记录而无明确的"不感兴趣"标签、医学诊断中仅有确诊病例而缺乏明确的阴性标本等。主流方法分为风险估计（nnPU、uPU）、标签消歧（DistPU）和辅助负样本选择（LaGAM）几大类。

**现有痛点**：即便是当前最好的 PU 方法，在复杂数据集上仍与全监督差距巨大——CIFAR-100 上最好的无辅助方法只有 76.49% OA，而监督学习可达 89.65%，差距超过 13 个点。作者通过 t-SNE 可视化直观展示了问题根源：LaGAM、HolisticPU 等方法学到的特征空间中正/负类分布严重重叠，而监督学习的特征可以清晰分离。这说明现有 PU 方法的根本瓶颈不在分类器设计、而在于无法从不可靠的伪标签中学到判别性表示。

**核心矛盾**：表示学习依赖准确的标签来构建同类/异类样本对，但 PU 场景下标签本身就不可靠——用伪标签构建的同类对中混杂大量"噪声对"（实际属于不同类但被错误认为同类的样本对），这些噪声对在标准对比/非对比损失下的梯度反而更大，主导了整个训练过程。形成恶性循环：差的表示→差的伪标签→更多噪声对→更差的表示。

**切入角度**：作者从两个关键观察出发。第一，非对比学习（只拉近同类、不推远异类）天然比对比学习更能容忍噪声标签，因为它不会错误地推远本应属于同类的样本。第二，标准非对比损失 $\mathcal{L}_r = 2(1 - \langle \tilde{q}_i, \tilde{k}_j \rangle)$ 的梯度与 $(1 - \cos^2\theta)$ 成正比——距离远（cos 小）的噪声对梯度大、距离近（cos 大）的 clean pair 梯度小，这恰好是反直觉的。而取 sqrt 后，梯度与 $(1 + \cos\theta)$ 成正比，距离近的 clean pair 梯度反而更大。

**核心 idea**：对标准非对比损失取 sqrt 翻转梯度-距离关系，让 clean pair 主导训练；结合 PhantomGate 提供保守负监督，形成 EM 式迭代互利框架。

## 方法详解

### 整体框架

NcPU 在 BYOL 非对比学习框架基础上构建。输入为正样本集 $\mathcal{P}$ 和无标签集 $\mathcal{U}$，每个样本经过随机增强产生两个视图，分别经 online network（含 encoder + projection head + prediction head）和 target network（momentum 更新，不含 prediction head）得到归一化嵌入 $\tilde{q}$ 和 $\tilde{k}$。分类器 $f(\cdot)$ 对每个样本输出二类 softmax 概率。整个训练流程有两个核心交替运行的模块：NoiSNCL 利用当前伪标签做噪声对鲁棒的类内表示对齐，PLD（含 PhantomGate）利用对齐后的表示空间更新更准确的伪标签。两者在理论上对应 EM 算法的 M-step 和 E-step。

### 关键设计

**1. NoiSNCL：给非对比损失套一个 sqrt，把训练主导权从噪声对夺回给 clean pair**

前面提到的恶性循环，根子在标准监督非对比损失 $\mathcal{L}_r = 2(1 - \langle \tilde{q}_i, \tilde{k}_j \rangle)$ 的梯度结构上——它对一个 pair 的梯度正比于 $(1 - \cos^2\theta)$，于是余弦相似度低、距离远的噪声对梯度大，而本应主导训练的 clean pair（余弦相似度高）梯度反而小，整个表示学习被噪声对牵着走。NoiSNCL 的改动只有一个：对损失开根号，$\tilde{\mathcal{L}}_r = 2\sqrt{1 - \langle \tilde{q}_i, \tilde{k}_j \rangle}$。这一步把梯度变成正比于 $(1 + \cos\theta)$，单调关系直接翻转——clean pair 梯度更大、噪声对被压制。直觉来自 $\sqrt{x}$ 的形状：它在 $x\to 0$ 附近导数趋于无穷（放大 clean pair 这种小 loss 对应的梯度），在 $x\to 1$ 附近导数趋于 0（抑制噪声对这种大 loss 的影响）。代价几乎为零：在干净的监督场景下 NoiSNCL 与标准损失性能相当（CIFAR-10 上 98.75% vs 98.53%），数值上也安全——BYOL 的非对称架构加随机增强保证 $\tilde{q}_i \neq \tilde{k}_j$，根号下不会取到 0。

**2. PhantomGate：用带 regret 的门控注入保守负监督，既防 trivial solution 又允许反悔**

PU 场景天生缺负类监督，如果只靠 prototype 相似度做消歧，模型很容易把所有样本都往正类拉，塌成 trivial solution（实验里这种做法 recall 98.7% 但 precision 仅 67%）；反过来简单加阈值硬选负样本又会矫枉过正，把准确率拉到 98% 却让 recall 崩到 0.51%。PhantomGate 走的是中间路线，整体分三步：先对类条件 prototype 每个 batch 做动量更新 $\mu_c = \text{Normalize}(\alpha \mu_c + (1-\alpha)\tilde{q})$；再用 prototype 相似度生成 soft 伪标签 $s'$ 并通过动量累积得到稳定估计；最后由门控决定每个样本的标签——当分类器给出的正类概率 $f_1(x) \geq \tau$ 时，直接把标签设成负类 $[0,1]^T$（这就是注入的保守负监督），否则退回到 prototype-based 的 $s'$。真正关键的是 regret 机制：一旦模型后来意识到某个被判负的样本可能错了，它不是从硬标签 $[0,1]^T$ 重新出发，而是从累积的 $s'$ 重新更新，从而避开了"一次误判就再也回不了头"的陷阱，让保守的负监督和纠错能力共存。

**3. 自适应阈值（SAT）：让负样本选择从松到紧自动调度，无需手调阈值**

PhantomGate 里那个判负的阈值 $\tau$ 并非固定值，而是由 SAT 在线维护：它跟踪一个全局阈值 $\tilde{\tau}$ 和一组类别感知调制因子 $\tilde{\rho}(c)$（都用动量更新），最终阈值取 $\tau = \frac{\tilde{\rho}(1)}{\max\{\tilde{\rho}(0), \tilde{\rho}(1)\}} \cdot \tilde{\tau}$。这套设计的好处是阈值会随训练自然演化——早期模型还不自信、$\tilde{\tau}$ 偏低，于是更多样本被选为负类、给出更密的监督信号；后期模型变自信、$\tilde{\tau}$ 升高，阈值随之收紧，把那些可能不准确的负类判断过滤掉。从松到紧的节奏正好对应课程学习的思路：先喂简单可靠的负监督，再逐步提高门槛。

### 损失函数 / 训练策略

总损失为三项之和：$\mathcal{L} = \frac{1}{|\mathcal{P}|}\sum_{x_i \in \mathcal{P}} \mathcal{L}_c + \frac{1}{|\mathcal{U}|}\sum_{x_i \in \mathcal{U}} \mathcal{L}_c + w_r \frac{1}{|\mathcal{D}|}\sum_{x_i \in \mathcal{D}} \frac{1}{|\mathcal{Q}|}\sum_{x_j \in \mathcal{Q}} \tilde{\mathcal{L}}_r$。其中 $\mathcal{L}_c$ 是标签消歧交叉熵（LDCE），$\tilde{\mathcal{L}}_r$ 是 NoiSNCL，$w_r = 50$ 控制表示学习的权重。所有动量超参 $\alpha = \beta = \gamma = 0.99$，五个数据集使用完全相同的超参数设置。Target network 采用 BYOL 式动量更新。训练过程中还使用了熵正则化稳定训练。Backbone 统一使用 ResNet-18。

### EM 理论框架

将分类器预测注入 EM 框架：E-step 对应伪标签分配（将每个无标签样本分配到正/负类簇），M-step 对应最小化 NoiSNCL（使簇内表示更紧凑）。Theorem 1 在 vMF 分布假设下证明：最小化 $\tilde{\mathcal{R}}_r$ 等价于最大化似然函数的一个下界 $L_1 = \sum_{\mathcal{S}_c} \frac{|\mathcal{S}_c|}{n_u} \|\nu_c\|^2 \leq L_2$。当 $\|\nu_c\| \to 1$（同类数据在表示空间高度聚集）时下界变紧。这为 NoiSNCL 和 PLD 的协同提供了原理性保证，而非仅仅是经验性组合。

## 实验关键数据

### 主实验

在 5 个数据集（3 个通用 + 2 个遥感灾损）上的对比，NcPU 在所有数据集上均取得最佳性能，且不依赖辅助信息：

| 方法 | 辅助信息 | CIFAR-10 OA | CIFAR-100 OA | STL-10 OA | ABCD OA | xBD OA |
|------|---------|-------------|-------------|-----------|---------|--------|
| CE（无标签当负样本） | 无 | 60.45 | 50.36 | 50.30 | 55.70 | 84.08 |
| uPU | $\pi_p$ | 65.52 | 61.44 | 57.08 | 83.76 | 86.82 |
| nnPU | $\pi_p$ | 87.29 | 72.00 | 80.62 | 87.73 | 82.60 |
| DistPU | $\pi_p$ | 85.29 | 67.63 | 85.62 | 86.25 | 82.94 |
| HolisticPU | 负样本 | 84.20 | 64.01 | 72.81 | 65.49 | 81.98 |
| LaGAM | 负样本 | 95.78 | 84.82 | 88.64 | 75.90 | 79.14 |
| WSC | 预估参数 | 90.55 | 75.39 | 79.06 | 80.10 | 84.89 |
| **NcPU** | **无** | **97.36** | **88.28** | **91.40** | **91.10** | **87.60** |
| Supervised | 全标签 | 96.96 | 89.65 | — | 92.00 | 88.47 |

注意 NcPU 在 CIFAR-10 上甚至超过了监督学习（97.36 vs 96.96），CIFAR-100 上差距仅 1.37%，ABCD 上差距不到 1%。

### 消融实验（CIFAR-100）

| 非对比损失 | 标签消歧 | OA | F1 | 说明 |
|-----------|---------|----|----|------|
| 无 | $s$（PhantomGate） | 61.54 | 40.58 | 无表示学习，仅靠标签消歧效果很差 |
| $\tilde{\mathcal{L}}_r$（NoiSNCL） | 无 | 50.27 | 1.09 | 无标签消歧，NoiSNCL 无法单独工作 |
| $\mathcal{L}_{self-r}$（自监督） | $s$ | 73.22 | 72.75 | 自监督非对比+PhantomGate |
| $\mathcal{L}_r$（标准监督） | $s$ | 84.58 | 85.90 | 标准损失已有效但被噪声对限制 |
| $\tilde{\mathcal{L}}_r$（NoiSNCL） | $s'$（仅 prototype） | 75.14 | 79.91 | 无 PhantomGate，precision 仅 67% |
| $\tilde{\mathcal{L}}_r$（NoiSNCL） | $s'$+SAT | 50.25 | 1.01 | SAT 引入的负监督太不准确 |
| $\tilde{\mathcal{L}}_r$（NoiSNCL） | $s$（PhantomGate） | **88.28** | **88.14** | 完整 NcPU |

### NoiSNCL 增强基础 PU 方法

| 方法 | CIFAR-10 OA | CIFAR-100 OA |
|------|-------------|-------------|
| uPU | 69.43 | 61.68 |
| uPU + $\tilde{\mathcal{L}}_r$ | **97.35** (+27.9) | **83.71** (+22.0) |
| nnPU | 83.25 | 71.22 |
| nnPU + $\tilde{\mathcal{L}}_r$ | **97.03** (+13.8) | **87.81** (+16.6) |
| Supervised + $\mathcal{L}_r$ | 98.53 | 94.45 |
| Supervised + $\tilde{\mathcal{L}}_r$ | 98.75 | 94.56 |

### 关键发现

- **NoiSNCL 是关键增益来源**：仅将 NoiSNCL 挂载到最简单的 uPU 上，CIFAR-10 就从 69.43%→97.35%（+27.9 个点），说明判别性表示才是 PU 学习的核心瓶颈，而非分类器设计
- **NoiSNCL vs 标准损失的差距**：在 CIFAR-100 上，$\tilde{\mathcal{L}}_r + s$（88.28%）比 $\mathcal{L}_r + s$（84.58%）高 3.7 个点，验证了噪声对鲁棒性的有效性；同时在监督学习下两者相当（98.75% vs 98.53%），说明 sqrt 变换没有引入额外代价
- **PhantomGate 的不可替代性**：单用 prototype 消歧（$s'$）导致 recall 高达 98.7% 但 precision 仅 67%（几乎全部标为正类）；加 SAT 后 precision 升到 98% 但 recall 跌至 0.5%（矫枉过正）；PhantomGate 的 regret 机制在两者间找到平衡（precision 89%, recall 87%）
- **超参不敏感**：所有 5 个数据集使用完全相同的超参（$\alpha=\beta=\gamma=0.99$, $w_r=50$），对 $\alpha$ 和 $\gamma$ 几乎不敏感，$\beta$ 越小伪标签更新越快，$w_r$ 越大表示学习越强
- **训练稳定**：CIFAR-10 上 400 epoch 后继续训练到 1200 epoch，OA 波动在 0.5% 以内，无过拟合或不稳定

## 亮点与洞察

- **sqrt 变换的巧妙性**：仅一个 sqrt 就翻转了梯度-距离的单调关系，从"噪声对主导"变为"clean pair 主导"。这个设计极其简洁却有深刻的数学直觉——$\sqrt{x}$ 在 $x\to 0$ 处导数趋无穷（放大小 loss 对应的 clean pair 的梯度），在 $x$ 较大时导数趋缓（抑制大 loss 对应的噪声对）。这种"改变损失形状以操纵梯度主导权"的思路可以迁移到任何噪声标签场景
- **理论和经验的闭环**：EM 理论不只是事后解释，它解释了为什么 NoiSNCL 和 PLD 必须联合使用——单独的 NoiSNCL（OA 50.27%）或单独的 PLD（OA 61.54%）都不行，但组合后达到 88.28%。E-step 提供更好的簇分配，M-step 让簇更紧凑，这个迭代互利在消融中被清晰验证
- **"简单方法+好表示" 的范式**：uPU + NoiSNCL（97.35%）超过了所有精心设计的 PU 方法，暗示当表示空间足够好时，最朴素的风险估计就够用了。这个洞察对整个弱监督学习社区都有启发

## 局限与展望

- **vMF 分布假设的局限**：EM 理论分析假设表示空间中每个类服从 vMF 分布（球面上的高斯），这对高度非球形分布的数据可能不成立。虽然实验表明即使假设不完全满足方法仍然有效，但理论保证可能不够紧
- **仅验证图像分类**：5 个数据集均为图像分类任务，NLP（如文本分类中的 PU learning）、图结构数据、表格数据上的效果未知。非对比学习在非视觉领域的增强效果可能不同
- **正样本数量固定**：实验中正样本数固定（CIFAR-10/100 用 1000 个），未分析正样本极度稀缺（如 <100 个）或相对充裕时的表现曲线
- **多分类扩展**：当前框架本质上是二分类（正 vs 负），如何扩展到多类 PU 学习（multiple positive classes + unlabeled）是一个开放问题
- **Backbone 的影响**：所有实验仅用 ResNet-18，更强的 backbone（如 ViT）或预训练特征是否会改变结论尚未探讨

## 相关工作与启发

- **vs LaGAM**：LaGAM 在 CIFAR-10（95.78%）和 CIFAR-100（84.82%）上是第二名，但需要辅助负样本作为输入。NcPU 在不使用任何辅助信息的情况下仍然超越（97.36% / 88.28%），且 LaGAM 在遥感数据（ABCD 75.90%）上表现很差，说明其泛化能力有限
- **vs DistPU**：DistPU 基于分布匹配做 PU 学习，在 STL-10 上有竞争力（85.62%），但依赖预估的类先验 $\pi_p$。NcPU 完全不需要 $\pi_p$ 且在所有数据集上都更好
- **vs WSC**：WSC 同样引入表示学习，但使用图论框架+对比学习+预估参数。NcPU 用更简单的非对比框架+EM 迭代取得更好效果，说明"噪声对鲁棒"比"更复杂的图结构"更重要
- **对噪声标签学习的启发**：NoiSNCL 的 sqrt 变换梯度反转思路可以直接借鉴到一般的噪声标签学习（Noisy Label Learning）中——任何需要从不可靠的 pair 关系中学表示的场景都可能受益

## 评分

- 新颖性: ⭐⭐⭐⭐ sqrt 变换翻转梯度主导权的想法简洁而深刻，PhantomGate 的 regret 机制也有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个数据集（含 2 个实际应用）+ 11 个 baseline + 详尽消融 + 超参分析 + 训练稳定性验证
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨清晰，梯度分析的可视化直观
- 价值: ⭐⭐⭐⭐⭐ 将 PU 学习性能提升到接近监督学习水平是该领域的里程碑式进展，NoiSNCL 的通用性超出 PU 学习本身

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Regulating Internal Alignment Flows for Robust Learning Under Spurious Correlations](regulating_internal_alignment_flows_for_robust_learning_under_spurious_correlati.md)
- [\[ICLR 2026\] Learning in Prophet Inequalities with Noisy Observations](learning_in_prophet_inequalities_with_noisy_observations.md)
- [\[ICLR 2026\] Scaling Direct Feedback Learning with Jacobian Alignment Guarantees](scaling_direct_feedback_learning_with_jacobian_alignment_guarantees.md)
- [\[ICLR 2026\] Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)
- [\[ICLR 2026\] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)

</div>

<!-- RELATED:END -->
