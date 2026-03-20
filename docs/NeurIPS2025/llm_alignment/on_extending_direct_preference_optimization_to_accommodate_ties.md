# On Extending Direct Preference Optimization to Accommodate Ties

**会议**: NeurIPS 2025  
**arXiv**: [2409.17431](https://arxiv.org/abs/2409.17431)  
**代码**: 无  
**领域**: LLM对齐 / 偏好优化  
**关键词**: DPO, preference optimization, ties, Bradley-Terry, Rao-Kupper, Davidson model

## 一句话总结
将 DPO 中的 Bradley-Terry 偏好模型替换为 Rao-Kupper 和 Davidson 扩展，使偏好优化能够显式建模"平局"数据，避免丢弃模糊偏好对，在翻译和数学推理上获得更好的正则化和性能。

## 研究背景与动机

1. **领域现状**：DPO 基于 Bradley-Terry 模型，要求每对训练数据必须有明确的胜负关系 $y_w \succ y_l$。实践中（如 Llama 3、Qwen2），会大量丢弃标注者难以区分优劣的"平局"数据。
2. **现有痛点**：丢弃平局数据是一种浪费——这些数据收集成本高，且确实包含有用信息（两者质量接近本身就是一种偏好信号）。直接将平局数据作为随机胜负塞入 DPO 会导致性能下降。
3. **核心矛盾**：Bradley-Terry 模型只有两个outcomes（$y_i$ 赢或 $y_j$ 赢），没有给"平局"留出概率空间。当 $\lambda_i \neq \lambda_j$ 时，模型永远偏向更强的一方。
4. **本文要解决什么？** 让 DPO 能正确利用平局数据，既不丢弃也不降低性能。
5. **切入角度**：从经典配对比较理论出发，Rao-Kupper (1967) 和 Davidson (1970) 早已提出了容纳平局的 Bradley-Terry 扩展——直接嵌入 DPO 框架即可。
6. **核心idea一句话**：用经典统计中的平局感知偏好模型替换 DPO 的 Bradley-Terry 模型，使优化目标对胜负对增大reward margin，对平局对驱动reward margin趋近零。

## 方法详解

### 整体框架
输入为带标记的偏好数据集：每对 $(x, y_w, y_l)$ 附带一个标志 $t \in \{0, 1\}$，$t=0$ 表示明确偏好，$t=1$ 表示平局。算法输出为训练后的策略 $\pi_\theta$。损失函数由两部分组成：对胜负对最大化偏好对数似然，对平局对最大化平局对数似然。

### 关键设计

1. **Rao-Kupper 模型 (DPO-RK)**:
   - 做什么：引入感知阈值 $\alpha_{RK}$，当reward差距小于阈值时视为平局
   - 核心思路：胜利概率 $p^{RK}(y_w \succ y_l) = \sigma(d_\theta - \alpha_{RK})$，相当于将 DPO 的sigmoid右移 $\alpha_{RK}$。平局概率由两端sigmoid的乘积给出。对胜负对，梯度推动 $d_\theta$ 增大；对平局对，梯度驱动 $d_\theta$ 趋近零
   - 设计动机：Rao-Kupper从"感知分辨率"出发，认为两者差距太小时评判者无法区分，这很符合偏好标注的实际情况

2. **Davidson 模型 (DPO-D)**:
   - 做什么：通过几何平均分配平局概率
   - 核心思路：$p^D(y_w \succ y_l) = \frac{1}{1 + e^{-d_\theta} + 2\nu_D e^{-d_\theta/2}}$。平局概率正比于两者强度的几何平均。满足 Luce 选择公理 $p(y_i \succ y_j)/p(y_j \succ y_i) = \lambda_i/\lambda_j$（Rao-Kupper不满足此性质）
   - 设计动机：Davidson从公理化角度推导，确保比较的逻辑一致性

3. **平局的正则化效应（理论解释）**:
   - 做什么：用理想 DPO 策略理论解释为什么平局数据能增强正则化
   - 核心思路：对于真实偏好概率 $\gamma = 0.5$ 的平局对，理想策略应满足 $\pi^*(y_w|x)/\pi^*(y_l|x) = \pi_{ref}(y_w|x)/\pi_{ref}(y_l|x)$，即策略不应偏离参考模型。这本质上是对 $\pi_\theta$ 的正则化约束——50%数据要求策略保持参考模型的行为
   - 设计动机：解释了为什么即使原始 DPO 加入平局也会出现更低的 KL 散度

### 损失函数 / 训练策略
$\mathcal{L}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{t=0}[\log p_\theta(y_w \succ y_l)] - \mathbb{E}_{t=1}[\log p_\theta(y_w \sim y_l)]$

超参数选择：$\nu_{RK} = 3$, $\nu_D = 1$，对应"同等实力的选手50%概率平局"的先验。实验表明结果对这些参数不敏感。

## 实验关键数据

### 主实验
三个任务：WMT21 ZH-EN 翻译、IWSLT17 FR-EN 翻译、TL;DR 摘要

| 方法 | WMT21 BLEURT | IWSLT17 BLEURT | TL;DR Win-Rate |
|------|-------------|----------------|----------------|
| DPO(CP) | 基线最优 | 基线最优 | 基线最优 |
| DPO(CP+TP) | 明显下降 | 明显下降 | 下降 |
| DPO-RK(CP+TP) | ≈DPO(CP)，KL更低 | ≈DPO(CP)，KL更低 | ≈DPO(CP)，KL更低 |
| DPO-D(CP+TP) | ≈DPO(CP)，KL更低 | ≈DPO(CP)，KL更低 | ≈DPO(CP)，KL更低 |

在数学推理（GSM8K/MATH）上的结果：
| 方法 | GSM8K Pass@1 | MATH Pass@1 |
|------|-------------|-------------|
| DPO | 81.4 | 44.1 |
| DPO-RK | **82.7** | **45.3** |
| DPO-D | 82.1 | 44.8 |

### 消融实验
| 配置 | 效果 | 说明 |
|------|------|------|
| 仅CP (原始DPO) | 高性能，高KL | 标准设置 |
| CP+TP (DPO) | 性能下降，低KL | 平局损害DPO |
| CP+TP (DPO-RK/D) | 高性能，低KL | 正确建模平局 |
| 平局比例增加 | KL进一步降低 | 正则化与平局比例成正比 |

### 关键发现
- DPO加入平局数据确实起到正则化作用（降低KL），但同时性能下降；DPO-RK/D保持性能不变的同时享受正则化效果
- 正则化效果与平局数据的比例成正比——翻译任务(50%平局)比TL;DR(12.5%平局)更明显
- Rao-Kupper和Davidson两个变体表现相当，对超参数不敏感
- 在数学推理任务上，利用奖励模型分数差距构造平局数据，DPO-RK 提升了1.3个百分点的GSM8K accuracy

## 亮点与洞察
- **巧妙的理论连接**：将70年代经典统计模型（Rao-Kupper 1967, Davidson 1970）与现代RLHF无缝对接。DPO-RK只需在sigmoid中加一个偏移 $\alpha_{RK}$，改动极小但效果显著
- **理想策略理论解释正则化**：通过Chen et al. (2024)的理想DPO策略理论严谨地解释了平局的正则化效应——不是经验观察而是理论推导
- **实用价值高**：对任何使用DPO训练的pipeline，只需将之前丢弃的"模糊"偏好对标记为平局，换用DPO-RK损失，就能免费获得数据利用率+正则化的双重收益

## 局限性 / 可改进方向
- 实验规模较小（翻译和TL;DR），未在大型LLM（>7B参数）上验证
- 仅考虑了两种经典模型，是否有更好的平局概率分配方案？
- 平局的定义依赖于启发式阈值（如BLEURT分差），更好的平局detection方法值得探索
- 没有与其他处理模糊偏好的方法（如soft labels, label smoothing）做对比

## 相关工作与启发
- **vs DPO**: DPO-RK/D是DPO的严格推广（$\alpha_{RK}=0$ 或 $\nu_D=0$ 退化为DPO），额外代价仅一个超参数
- **vs IPO**: IPO也试图处理偏好的不确定性，但通过替换sigmoid为平方损失实现，没有显式建模平局。实验表明IPO加入平局后也有类似问题
- **vs KTO**: KTO可以利用非配对数据，但仍需二元标签；本文的方法保持配对框架但增加了第三种标签

## 评分
- 新颖性: ⭐⭐⭐⭐ 经典统计模型+现代DPO的结合非常优雅，但从技术角度看并不复杂
- 实验充分度: ⭐⭐⭐⭐ 三个任务+数学推理，有理论分析支撑，但缺乏大模型实验
- 写作质量: ⭐⭐⭐⭐⭐ 论文写得清晰流畅，动机、理论、实验环环相扣
- 价值: ⭐⭐⭐⭐ 高度实用的改进，几乎零成本采用
