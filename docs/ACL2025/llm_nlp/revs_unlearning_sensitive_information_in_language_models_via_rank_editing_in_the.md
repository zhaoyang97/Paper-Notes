# REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space

**会议**: ACL 2025  
**arXiv**: [2406.09325](https://arxiv.org/abs/2406.09325)  
**代码**: https://github.com/tomerashuach/REVS (有)  
**领域**: LLM安全 / 机器遗忘  
**关键词**: 机器遗忘, 词汇空间编辑, 排名操作, 无梯度, 敏感信息保护

## 一句话总结

提出 REVS，一种无梯度的模型编辑方法，通过在 FF2 层中定位与敏感 token 关联最强的神经元，将其投影到词汇空间后迭代降低目标 token 排名，在 SSN/Email/URL 三类敏感数据上 Unlearning Score 显著超越 6 种基线（89.58 vs 36.98），同时通用能力几乎零损（MMLU 61.05→60.87），且对 Logit-Lens 和 Delta 提取攻击高度鲁棒。

## 研究背景与动机

1. **领域现状**：LLM 会记忆训练数据中的敏感信息（邮箱、URL、身份证号等），给定相关 prompt 即可还原完整 PII。现有防御分两类——数据侧的差分隐私/数据清洗（需重训练），模型侧的机器遗忘/模型编辑（后处理修改参数）。
2. **现有痛点**：(1) 差分隐私降低模型质量，数据清洗代价高且每次发现新 PII 都要重训练；(2) ICL 遗忘和梯度上升（GA）只是表面抑制生成，参数中仍保留信息，对白盒提取攻击脆弱；(3) MEMIT 等模型编辑方法本来是为"修改事实"设计的，用于"遗忘"时效果差；(4) 现有 PII 遗忘评估缺少真正包含自然记忆敏感信息的数据集。
3. **核心矛盾**：梯度方法改动范围大容易损害通用能力，而精准编辑方法又不是为遗忘目标设计的。
4. **切入角度**：Transformer MLP 的 FF2 层相当于"记忆存储"（Geva et al., 2021），其每一列（神经元）投影到词汇空间后对应具体 token 的 logit 贡献。如果能精准找到促进敏感 token 的神经元，直接在词汇空间降低排名，就无需梯度计算。
5. **核心 idea 一句话**：不优化损失函数，而是在词汇空间中直接操作 token 排名——把敏感 token 从 rank 1 降到指定低排名。

## 方法详解

### 整体框架

输入：模型 $M$、包含敏感信息的 prompt $P$、目标 token $t$、目标排名 $r_d$、最大编辑神经元数 $n_{max}$。输出：编辑后的模型，$t$ 在 $P$ 对应位置排名降至 $r_d$ 以下。整个流程分两个阶段：**定位**（找到哪些层/神经元负责 $t$）→ **编辑**（在词汇空间中调降 $t$ 的 logit 排名）。在编辑前还有一个**目标 token 选择**步骤，只选序列中最稀有的 2 个 token 作为遗忘目标。

### 关键设计

1. **层选择（Layer Selection）**
   - 做什么：从所有 Transformer 层中筛选与目标 token 关联强的层
   - 核心思路：对每层 $l$，将 FF2 的输出 $\vec{h}_l$ 经 unembedding 矩阵 $U$ 投影到词汇空间，计算目标 token $t$ 的排名 $r_{t,l} = \text{rank}(t, U\vec{h}_l)$；排名低于阈值 $r_h$ 的层被选中
   - 设计动机：排名越高（数值越小），说明该层对生成 $t$ 的贡献越大，只编辑这些层可最小化对模型的扰动

2. **神经元选择（Neuron Selection）**
   - 做什么：在选中的层内，找到与 $t$ 关联最强的 FF2 列（神经元）
   - 核心思路：两步混合筛选——(a) 先取激活值 $a_j$ 最高的 top-$k$ 个神经元（上下文相关性）；(b) 在这 $k$ 个中按 $\text{rank}(t, U\vec{n}_j)$ 排序，逐步选取直到整层 $r_t > r_h$ 或达到 $n_{max}$
   - 设计动机：仅按激活或仅按 token 关联单独选都不够好。激活保证上下文相关（该神经元确实被当前 prompt 激活），token 关联保证语义相关（该神经元确实编码了目标 token）。消融实验验证混合策略 Unlearning Score 75.2，优于纯激活（68.8）、纯排名（22.4）、梯度（71.1）和随机（0）

3. **神经元编辑（Neuron Editing）**
   - 做什么：对选中的神经元 $\vec{n}_j$，在词汇空间中迭代调整直到 $t$ 排名降至 $r_n$
   - 核心思路：(a) 投影到词汇空间 $\vec{v} = U\vec{n}_j$；(b) 将 $t$ 的 logit 设为低值 $l_t$（初始 $-10$）；(c) 用伪逆 $U^\dagger$ 投影回神经元空间 $\vec{n}_j^* = U^\dagger \vec{v}$；(d) 再投影回词汇空间检查 $t$ 排名。因为 $U$ 不可逆，伪逆引入近似误差，所以需要迭代：排名不够低则 $l_t \leftarrow l_t \times 1.3$，太低则 $l_t \leftarrow l_t \times 0.8$，直到收敛到 $\epsilon$ 容差内
   - 设计动机：直接置零神经元（DEPN 方式）会破坏该神经元对其他 token 的贡献，导致 specificity 下降。REVS 只改变 $t$ 的排名，保留其他 token 的 logit 分布

4. **目标 token 选择（Target Token Selection）**
   - 做什么：从敏感序列 $S$ 中选最少的 token 子集 $T \subseteq S$ 来遗忘
   - 核心思路：选 $|T|=2$ 个最稀有的 token（用 token ID 近似稀有度，ID 越大越稀有），排除常见部分如 `@email.com`、`http`、`://` 等
   - 设计动机：恢复完整敏感信息需要还原全部 token 序列，只要其中 2 个关键稀有 token 无法生成，整条序列就不可恢复。这大幅减少编辑量

### 训练策略

REVS 完全无梯度。在每个实验中，先对所有敏感序列执行遗忘编辑，再统一评估。超参数通过在 SSN 和 Email 数据集上最大化 Unlearning Score（k=100）来优化。Llama-3-8B 上每层最多编辑 130 个神经元（SSN）/ 45 个（Email），激活 top-k=1000，目标排名 $r_n = 105000$（词汇表大小 128,256）。

## 实验关键数据

### 主实验（Llama-3-8B, k=100）

| 方法 | Unlearning Score ↑ | Efficacy@100 ↑ | Generality@100 ↑ | Specificity ↑ | MMLU | GSM8K |
|------|---:|---:|---:|---:|---:|---:|
| FT-L | 36.98 | 63.88 | 50.35 | 24.33 | 60.99 | 46.62 |
| MEMIT | 24.72 | 30.70 | 23.90 | 22.67 | 61.02 | 46.17 |
| NPO-KL | 11.95 | 38.78 | 36.13 | 6.33 | 61.01 | 47.23 |
| RMU | 16.42 | 13.47 | 16.67 | 38.67 | 60.83 | 48.21 |
| Max-Entropy | 5.12 | 5.17 | 3.92 | 1.40 | 61.06 | 47.46 |
| Head-Projection | 2.98 | 3.08 | 2.95 | 4.17 | 61.06 | 46.92 |
| **REVS** | **89.58** | **98.88** | **89.67** | **82.17** | 60.87 | 44.20 |

*SSN 数据集结果。Unlearning Score 是 Efficacy、Specificity 和 Generality 的调和平均。REVS 统计显著优于所有基线（Wilcoxon signed-rank, p<0.05）。*

### 提取攻击抵抗（Llama-3-8B, SSN, k=100）

| 方法 | Resistance Score ↑ | Logit-Lens@100 ↑ | Delta@100 ↑ | Perturb@100 ↑ |
|------|---:|---:|---:|---:|
| FT-L | 82.77 | 63.88 | 98.08 | 98.22 |
| MEMIT | 55.20 | 30.70 | 97.90 | 98.18 |
| NPO-KL | 61.63 | 38.78 | 98.47 | 95.08 |
| **REVS** | **99.27** | **98.88** | **98.92** | **100.00** |

*REVS 在三种攻击下均几乎完美防御，Resistance Score 99.27 远超第二名 FT-L 的 82.77。*

### 消融实验

**神经元选择策略消融（GPT-J-6B, Email）**

| 选择方法 | Unlearning Score | Resistance Score |
|----------|---:|---:|
| Random | 0 | 0 |
| Rank only | 22.4 | 29.2 |
| Activation only | 68.8 | 81.8 |
| Gradient-based | 71.1 | 78.2 |
| **Rank & Activation (hybrid)** | **75.2** | **84.2** |

**编辑策略消融：置零 vs 排名编辑（GPT-J-6B, Email）**

| 方法 | Unlearning Score | Efficacy@100 | Specificity | Resistance Score |
|------|---:|---:|---:|---:|
| ZERO (n=5) | 0 | 73.8 | 0 | 79.2 |
| ZERO (n=2) | 14.3 | 26.3 | 9.7 | 47.5 |
| ZERO (n=1) | 11.2 | 6.8 | 31.9 | 17.3 |
| **REVS** | **83.5** | **81.1** | **87.1** | **82.6** |

**目标 token 选择策略消融（GPT-J-6B, Email）**

| Token 选择 | Unlearning Score | Efficacy@100 | Specificity | Resistance Score |
|-----------|---:|---:|---:|---:|
| Most Frequent | 55.6 | 67.7 | 47.22 | 70.6 |
| First | 65.6 | 86.8 | 52.77 | 77.2 |
| Random | 72.0 | 89.0 | 60.4 | 80.5 |
| **Rarest** | **75.2** | **96.1** | **61.8** | **83.9** |

### 关键发现

- **REVS 在三类数据上全面领先**：SSN 上 Unlearning Score 89.58（最近基线 FT-L 36.98，差距 2.4×），Email 62.37，URL 44.25，均为最优
- **提取攻击几乎完美防御**：SSN 上 Resistance Score 99.27，Perturb Attack 100.00；URL 上 82.80 也显著超越
- **通用能力几乎无损**：MMLU 从 61.05 降到 60.87（-0.3%），GSM8K 在 SSN 上有所下降（47.83→44.20），作者认为因 SSN 是数字型数据
- **混合神经元选择的必要性**：纯排名选择 Unlearning Score 仅 22.4，纯激活 68.8，混合达 75.2——单一信号不够
- **置零不可行**：直接置零神经元导致 specificity 崩溃（ZERO n=5 的 specificity 为 0），说明神经元服务于多个 token
- **选最稀有 token 最有效**：2 个最稀有 token 就够遗忘整条序列，且好于频率最高/首个/随机选择
- **GPT-J-6B 上相同趋势**：REVS 在 SSN 81.45、Email 80.65 同样领先，验证方法通用性
- **Pareto 优势**：REVS 在 efficacy-specificity 权衡曲线上 Pareto 支配 FT-L，尤其在 SSN 上几乎无权衡

## 亮点与洞察

- **"排名而非值"的编辑哲学**：不需要精确调整 logit 绝对值，只需让目标 token 排名足够低。这绕开了精确控制 logit 的困难，且对排名变化的容忍度更高——只要 $t$ 不在 top-k 内即可。这个思路可迁移到任何需要抑制特定输出的场景（如有毒内容、版权文本）
- **伪逆迭代的工程智慧**：因为 unembedding 矩阵 $U$ 不是方阵、不可逆，用伪逆 $U^\dagger$ 投影回去会有误差。作者没有试图找更好的逆、而是用简单的迭代乘法因子（1.3/0.8）收敛。实用且稳健
- **"遗忘 2 个 token 即可消除整条序列"**：关键洞察是信息恢复需要完整序列，因此只需破坏最难猜测的 2 个稀有 token。这大幅减少了对模型的修改量，是 specificity 高的根本原因
- **自然记忆数据集的构建方法**：利用 Pile 的 extraction benchmark 子集 + Presidio PII 检测 + 模型生成验证，构建了真实记忆的 Email/URL 数据集。这个流程本身对后续 PII 遗忘研究有参考价值

## 局限性 / 可改进方向

- **规模有限**：仅在 6B/8B 模型验证，未测试 70B+ 模型。大模型中信息可能更分布式
- **数据类型有限**：主要测邮箱、URL 和 SSN（数字），对电话号码、地址等未验证
- **SSN 上 GSM8K 下降**：数字型 PII 遗忘影响了数学推理（47.83→44.20），说明数字 token 的编辑更难做到 specific
- **仅英文**：未在其他语言上评估
- **攻击模型有限**：只考虑了 Logit-Lens、Delta 和 Perturbation 三种攻击，未评估 probe-based 或注意力头级别的提取
- **可改进**：(1) 结合 attention head 分析做更精准的定位；(2) 对数字型 token 设计专门的选择策略避免通用能力损害；(3) 探索编辑后能否在不损害遗忘效果的前提做轻量微调恢复 GSM8K

## 相关工作与启发

- **vs MEMIT (Meng et al., 2023)**：MEMIT 是知识编辑方法，通过计算约束优化 FF2 来插入/修改事实。本文将其改为遗忘目标，但效果有限（Unlearning Score 24.72 vs REVS 89.58），因为 MEMIT 不是为排名操作设计的。REVS 的排名视角更适合遗忘任务
- **vs FT-L (Zhu et al., 2020)**：FT-L 通过梯度上升 + $L_\infty$ 约束微调 FF2。是梯度方法中最强的基线（Unlearning Score 36.98），但 specificity 差（24.33），且随目标数增加性能不稳定。REVS 的无梯度方式在稳定性和 specificity 上都更优
- **vs DEPN (Wu et al., 2023)**：最相似的工作，也定位神经元，但直接置零。消融实验表明置零导致 specificity 崩溃。REVS 的"降排名而非置零"是关键改进
- **vs RMU (Li et al., 2024)**：RMU 在表示空间做方向扰动，面向 WMDP 这类概念遗忘。用于 PII 遗忘时效果差（Unlearning Score 16.42），因为 PII 是 token 级问题而非概念级

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 词汇空间排名编辑是全新思路，将遗忘转化为排名操作而非值优化
- 实验充分度: ⭐⭐⭐⭐⭐ 6 基线 + 3 数据集 + 2 模型 + 3 种攻击 + 4 组消融，覆盖极为全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码完整，但附录较长且部分结果分散
- 价值: ⭐⭐⭐⭐⭐ 对 GDPR 合规的 PII 遗忘有直接实用价值，且构建了首个自然记忆敏感信息数据集
