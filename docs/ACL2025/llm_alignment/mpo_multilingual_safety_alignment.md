# MPO: Multilingual Safety Alignment via Reward Gap Optimization

**会议**: ACL 2025  
**arXiv**: [2505.16869](https://arxiv.org/abs/2505.16869)  
**代码**: [https://github.com/circle-hit/MPO](https://github.com/circle-hit/MPO)  
**领域**: 对齐RLHF  
**关键词**: multilingual safety, reward gap, preference optimization, cross-lingual transfer, alignment  

## 一句话总结
MPO 发现 LLM 在主导语言（英文）和目标语言间的隐式 Reward Gap 与安全性能强相关，提出直接最小化两者 Reward Gap 差异来将主导语言的安全对齐能力迁移到多语言，在三个模型上显著降低了低资源语言的攻击成功率且不损害通用能力。

## 研究背景与动机
1. **领域现状**：LLM 安全对齐（RLHF/DPO）主要在英文上进行，但全球用户使用多种语言，低资源语言的安全性严重不足
2. **现有痛点**：
   - 多语言偏好数据稀缺，翻译数据质量差（尤其低资源语言存在不自然表述和内容错误）
   - DPO/RLHF 对噪声数据高度敏感，噪声多语言数据可能导致安全性错误对齐（safety misalignment）
   - 直接在各语言上做偏好学习效果有限，模型在 Bengali、Swahili 等语言上攻击成功率高达 55-98%
3. **核心矛盾**：主导语言已有良好的安全对齐，但这些能力无法自动迁移到其他语言
4. **本文要解决什么？** 如何利用主导语言已有的安全能力来提升目标语言的安全性
5. **切入角度**：发现 Reward Gap（safe 和 unsafe 回复的对数似然差）与安全性能（ASR）呈强负相关，将其作为跨语言安全迁移的桥梁
6. **核心idea一句话**：最小化目标语言与主导语言间的 Reward Gap 差距，无需直接优化噪声多语言偏好数据

## 方法详解

### 整体框架
MPO（Multilingual reward gaP Optimization）分两步：(1) 计算主导语言（英文）在参考模型上的 Reward Gap $\text{RG}^d$（作为固定目标），(2) 优化训练模型使目标语言的 $\text{RG}^t$ 趋近 $\text{RG}^d$，并约束主导语言的隐藏表示不变化。输入是平行的多语言偏好数据（英文原始 + 翻译版），输出是多语言安全对齐的模型。

### 关键设计

1. **Reward Gap 定义（基于 SimPO 长度归一化）**:
   - 做什么：量化模型对 safe vs unsafe 回复的区分度
   - 核心思路：$\text{RG}^t = \frac{1}{|y_w^t|}\log\pi_\theta(y_w^t|x^t) - \frac{1}{|y_l^t|}\log\pi_\theta(y_l^t|x^t)$，采用 SimPO 的平均对数似然而非 DPO 的 log-ratio
   - 设计动机：(a) 与推理时的 likelihood 度量一致，RG 越大意味着生成安全回复的概率越高；(b) 长度归一化消除了 unsafe 回复通常较长（包含具体有害内容）vs safe 回复简短（拒绝模式）的偏差
   - 实验验证：在 LLaMA-3.1、Gemma-2、Qwen2.5 上，英文 RG 分别为 1.58/2.32/1.87，Swahili 仅 0.05/0.41/0.20，与 ASR 呈强负相关

2. **Reward Gap 差异最小化（$\mathcal{L}_1$）**:
   - 做什么：让目标语言的 RG 趋近主导语言的 RG
   - 核心思路：$\mathcal{L}_1 = \mathbb{E}[\|\beta \cdot \text{RG}^t - \text{RG}^d\|^2]$，其中 $\text{RG}^d$ 由参考模型（原始对齐模型）计算并固定，$\text{RG}^t$ 由训练模型在目标语言上计算
   - 设计动机：不直接优化噪声多语言偏好数据的 pairwise loss，而是以主导语言 RG 为"锚"来间接传递安全能力。梯度分析显示 $w_\theta = \beta\text{RG}^t - \text{RG}^d$ 自适应调节更新幅度和方向——RG 差距大时更新更强烈
   - 与 DPO 的区别：DPO 的梯度权重基于模型自身 likelihood，受数据噪声影响大；MPO 的权重基于 RG 差距，由高质量主导语言信号引导

3. **主导语言表示保护（$\mathcal{L}_2$）**:
   - 做什么：防止多语言对齐过程中损害主导语言的已有能力
   - 核心思路：$\mathcal{L}_2 = \mathbb{E}[\|\mathbf{h}^d - \mathbf{h}^d_\text{ref}\|^2]$，约束主导语言 last token 的隐藏表示与参考模型保持一致
   - 设计动机：直接约束隐藏表示比 KL 散度正则化（logit 层面）更有效——近期研究表明修改隐藏表示对行为控制更直接。避免了"拆东墙补西墙"

### 损失函数 / 训练策略
- 最终损失：$\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_2$
- 训练数据：PKU-SafeRLHF 英文偏好数据 + Google Translate 翻译到 5 种目标语言
- 不需要额外的多语言人工标注——翻译数据虽有噪声，但 MPO 不直接优化偏好对，而是优化 RG 差距，对噪声更鲁棒

## 实验关键数据

### 主实验

**MultiJail 安全评估（ASR↓，LLaMA-3.1-8B）**:

| 方法 | En | Zh | Ko | Ar | Bn | Sw | AVG. |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:----:|
| 原始模型 | 14.6 | 20.3 | 52.4 | 16.8 | 49.5 | 37.8 | 31.9 |
| SFT | 12.7 | 9.8 | 31.4 | 8.6 | 31.8 | - | - |
| DPO | - | - | - | - | - | - | ~25 |
| SimPO | - | - | - | - | - | - | ~23 |
| **MPO** | - | - | - | - | - | - | **~15** |

**跨模型泛化（AdvBench-X AVG. ASR↓）**:

| 模型 | 原始 | DPO | SimPO | MPO |
|------|:----:|:---:|:-----:|:---:|
| LLaMA-3.1-8B | 20.86 | ~15 | ~14 | **~8** |
| Gemma-2-9B | ~8 | ~6 | ~5 | **~3** |
| Qwen2.5-7B | ~30 | ~22 | ~20 | **~12** |

### 消融实验

| 配置 | MultiJail AVG | AdvBench-X AVG | 说明 |
|------|:------------:|:-------------:|------|
| Full MPO | 最优 | 最优 | $\mathcal{L}_1 + \mathcal{L}_2$ |
| w/o $\mathcal{L}_2$ | +3-5% | +2-4% | 主导语言性能下降 |
| w/o length norm | +2-3% | +1-2% | 长度偏差影响 |
| 用 DPO-style RG | +1-2% | +1-2% | SimPO RG 更优 |

### 关键发现
- RG 与 ASR 呈强负相关：英文 RG 高（1.58-2.32）→ ASR 低（0-13%），低资源语言 RG 低（0.04-0.20）→ ASR 高（55-98%）
- MPO 对翻译数据噪声更鲁棒：在使用 Google Translate 翻译的（噪声较大的）数据上，MPO 比 DPO/SimPO 优势更大——因为 MPO 不直接优化偏好对，而是优化 RG 差距
- 不损害通用能力：在 MT-Bench、mGSM 等通用benchmark上，MPO 不降低甚至略微提升多语言通用性能
- 表示保护 $\mathcal{L}_2$ 对保持主导语言性能至关重要

## 亮点与洞察
- **Reward Gap 作为安全性的可量化代理指标**：这个发现本身很有价值——提供了一种无需输出评估就能衡量 LLM 安全对齐质量的方法，可用于监控生产环境中的安全状态
- **以主导语言为锚的对齐迁移**：不直接优化噪声翻译数据的偏好对，而是以主导语言 RG 为目标的间接优化，巧妙地绕过了数据质量问题。这个思路可以推广到其他跨语言能力迁移场景
- **SimPO 式 RG 优于 DPO 式 RG**：长度归一化对多语言安全场景特别重要，因为拒绝回复（safe）通常远短于违规回复（unsafe）

## 局限性 / 可改进方向
- **依赖翻译数据**：虽然 MPO 对翻译噪声更鲁棒，但仍需平行偏好数据作为输入。可探索完全无翻译的方案（如直接在目标语言上用 RG 作为自监督信号）
- **主导语言假设**：假设存在一个已良好对齐的主导语言，对于所有语言安全性都差的模型不适用
- **文化差异未处理**：不同语言/文化对"安全"的定义可能不同，简单迁移英文安全标准可能不适配
- **仅测试了 8-9B 模型**：在更大模型上的效果未验证

## 相关工作与启发
- **vs DPO/SimPO**: DPO/SimPO 直接优化偏好对的 pairwise loss，对噪声翻译数据敏感；MPO 优化 RG 差距，更鲁棒
- **vs 多语言 RLHF**: 传统多语言 RLHF 需要为每种语言训练 reward model 或收集偏好数据；MPO 复用主导语言的 RG 信号
- **vs representation engineering**: MPO 的 $\mathcal{L}_2$ 与 representation engineering 思路一脉相承——通过控制隐藏表示来保护能力
- 可以探索将 RG 迁移思路应用于其他对齐维度（如 helpfulness 跨域迁移）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Reward Gap 作为跨语言安全迁移的桥梁是非常新颖的视角
- 实验充分度: ⭐⭐⭐⭐ 三个模型、六种语言、多个安全benchmark、消融完整
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，数学推导严谨，梯度分析有深度
- 价值: ⭐⭐⭐⭐⭐ 多语言安全是关键问题，MPO 方案简洁有效且不需额外标注成本
