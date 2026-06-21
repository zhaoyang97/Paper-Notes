---
title: >-
  [论文解读] Influence-Preserving Proxies for Gradient-Based Data Selection in LLM Fine-Tuning
description: >-
  [ICLR 2026][LLM效率][梯度数据选择] IPROX 不再用现成小模型当代理来做梯度影响力数据选择，而是从目标 LLM 直接"蒸出"一个保留影响力信息的低秩代理——先用影响力加权的 SVD 压缩、再用梯度对齐微调，使得一个更小的代理在选数据时甚至胜过更大的现成代理。 领域现状：监督微调（SFT）的效果高度依赖训练…
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "梯度数据选择"
  - "Influence Function"
  - "TracIn"
  - "代理模型"
  - "低秩压缩"
  - "SVD"
  - "监督微调"
---

# Influence-Preserving Proxies for Gradient-Based Data Selection in LLM Fine-Tuning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PDNpRLxDlI](https://openreview.net/forum?id=PDNpRLxDlI)  
**代码**: [https://github.com/csr16/IProX](https://github.com/csr16/IProX)  
**领域**: LLM 高效化 / 数据选择  
**关键词**: 梯度数据选择, Influence Function, TracIn, 代理模型, 低秩压缩, SVD, 监督微调  

## 一句话总结
IPROX 不再用现成小模型当代理来做梯度影响力数据选择，而是从目标 LLM 直接"蒸出"一个保留影响力信息的低秩代理——先用影响力加权的 SVD 压缩、再用梯度对齐微调，使得一个更小的代理在选数据时甚至胜过更大的现成代理。

## 研究背景与动机
**领域现状**：监督微调（SFT）的效果高度依赖训练数据的选择，盲目堆数据反而会损害下游性能。梯度数据选择（gradient-based data selection）是当前的主流路线，它"模型感知"地用样本梯度来估计每个样本对验证性能的贡献——代表方法是 TracIn（沿训练轨迹累加训练样本与验证样本的梯度内积）和 Influence Function（用逆 Hessian 缩放样本梯度，近似 leave-one-out 效应）。

**现有痛点**：这两类方法的计算开销极高，要么需要存大量 checkpoint 反复反传，要么需要算昂贵的逆 Hessian 向量积，开销随模型规模急剧膨胀，对几十亿参数的 LLM 基本不可行。一个常见的折中是用**现成的小模型当代理**（如用 Llama3-8B 帮 Llama3-70B 选数据），但论文指出现成代理有三个硬伤：① 它的学习动态不清楚，选谁当代理只能靠"大模型行为像小模型"这种先验假设；② 同一模型家族只有少数几个固定尺寸，无法按算力预算灵活调代理大小；③ **没有系统办法把代理与目标模型在"影响力估计"这件事上对齐**。

**核心矛盾**：代理要足够小才省算力，又要足够"像"目标模型才能选对数据——现成代理在这两端都不可控。

**本文目标**：直接从目标模型构造一个尺寸可调、且**显式保留目标模型梯度影响力**的代理，让影响力计算被卸载到这个便宜代理上。

**核心 idea**：**与其挑一个假设偏好的小模型，不如从目标模型本身派生一个继承其梯度特性的小模型。** 关键洞察是——标准 SVD 压缩最小化的是权重重构误差（Frobenius 范数），但这和"保留影响力"目标错位（实验显示压缩稀疏度一升高，影响力保持率比损失保持率掉得快得多）。因此需要一种**显式以保留影响力为目标**的压缩。

## 方法详解

### 整体框架
IPROX 是一个两阶段框架：**阶段一**用"影响力保持 SVD（IPSVD）"把目标模型逐层低秩压缩，得到一个尺寸可控且保留影响力信息的初始代理；**阶段二**通过梯度对齐 + logits 锚定进一步精修这个代理，补偿逐层压缩累积的误差。最终用这个代理算影响力分数、选 Top-k 数据去微调目标模型。

```mermaid
flowchart LR
    A[目标 LLM fθ<br/>warm-up 后] --> B[阶段一 IPSVD<br/>影响力加权低秩压缩]
    B --> C[初始代理 fθ′<br/>W≈AB 低秩]
    C --> D[阶段二 对齐]
    D -->|内部: 低秩空间梯度对齐| E[精修代理]
    D -->|外部: KL logits 锚定| E
    E --> F[在代理上算 TracIn/IF 影响力<br/>选 Top-k 数据]
    F --> G[微调目标模型]
```

### 关键设计
**1. 影响力保持的 SVD（IPSVD）：用二阶矩重加权对齐"影响力"而非"重构误差"。** 问题出在标准 SVD 的目标函数：它给的是 Frobenius 重构误差下的最优低秩近似，但这不保证压缩后代理还能保留梯度影响力。论文先做了一个理论铺垫：对某层权重 $W_\ell$ 的影响力可写成 $I_{W_\ell}(z,z')=\langle\delta_\ell(z),\delta_\ell(z')\rangle_F\,\langle h_{\ell-1}(z),h_{\ell-1}(z')\rangle_F$（其中 $h_{\ell-1}$ 是层输入、$\delta_\ell$ 是上游梯度），一个小扰动 $E_\ell$ 对影响力的影响主要通过局部方向效应 $e_\ell(z)\triangleq\delta_\ell(z)^\top E_\ell\, h_{\ell-1}(z)$ 体现。命题 4.1 证明：在局部光滑等假设下，影响力的期望变化被 $\sqrt{\mathbb{E}_z[e_\ell(z)^2]}$ 上界控制。于是"保留影响力"就转化为最小化 $\mathbb{E}_z[e_\ell(z)^2]$，在 K-FAC 近似下它等价于一个**加权 Frobenius 范数**：

$$\min_{\widehat{W}_\ell}\ \big\|\,C_{\delta,\ell}^{1/2}(W_\ell-\widehat{W}_\ell)\,C_{h,\ell}^{1/2}\,\big\|_F^2$$

其中 $C_{h,\ell}=\mathbb{E}[h_{\ell-1}h_{\ell-1}^\top]$、$C_{\delta,\ell}=\mathbb{E}[\delta_\ell\delta_\ell^\top]$ 是输入与上游梯度的二阶矩。这两个矩阵实际起的是**重加权**作用——在输入幅值大、损失最敏感的方向上更重地惩罚误差，从而优先保留对影响力最关键的权重分量。具体做法是对重加权矩阵 $S_\ell=C_{\delta,\ell}^{1/2}W_\ell C_{h,\ell}^{1/2}$ 做 SVD 并截断到 top-$r_\ell$，再变换回原空间得到 $\widehat{W}_\ell=A_\ell B_\ell$，秩 $r_\ell$ 直接决定代理尺寸。

**2. 用"瘦 SVD"绕开大矩阵，让压缩本身也廉价。** 直接构造并求 $C_{h,\ell}$、$C_{\delta,\ell}$ 的平方根与逆，对大模型来说代价高得离谱（$O(n_\ell^3+m_\ell^3)$）。IPROX 用一个只有 $N$ 个样本的小探针集（probe set），一次前向+反向就收集各层的输入矩阵 $H_\ell\in\mathbb{R}^{n_\ell\times N}$ 和梯度矩阵 $\Delta_\ell\in\mathbb{R}^{m_\ell\times N}$，对这两个"高瘦"矩阵直接做 skinny SVD，再用它们拼出一个至多 $N\times N$ 的小核矩阵的 SVD——把复杂度降到 $O(N^3+n_\ell N^2+m_\ell N^2)$（$N\ll n_\ell,m_\ell$）。这是 IPROX 在实践中能"几分钟构造代理"的关键。

**3. 低秩空间内的梯度对齐：对齐影响力又不破坏效率。** 阶段一的初始代理满足命题 4.1 的界，但逐层近似误差会累积，需要进一步对齐。直觉做法是把代理梯度重构回目标模型的高维权重空间再比对，但这样一来后续每算一次影响力都要做高维重构，效率优势全没了。IPROX 反过来做：把**目标模型的梯度投影下到低秩代理空间**里对齐。因为代理层是 $W_\ell\approx A_\ell B_\ell$，用链式法则把 $\nabla_{W_\ell}L$ 投到 $A_\ell,B_\ell$ 上（$\nabla_{A_\ell}L=\nabla_{W_\ell}L\,B_\ell^\top$、$\nabla_{B_\ell}L=A_\ell^\top\nabla_{W_\ell}L$），得到对齐损失：

$$\mathcal{L}_{GA}=\frac{1}{|L|}\sum_{\ell\in L}\Big(d\big(\nabla_{A_\ell}L,\ \mathrm{sg}(\nabla_{W_\ell}L)B_\ell^\top\big)+d\big(\nabla_{B_\ell}L,\ A_\ell^\top\mathrm{sg}(\nabla_{W_\ell}L)\big)\Big)$$

其中 $\mathrm{sg}(\cdot)$ 是 stop-gradient。对齐完全在代理参数空间内完成，影响力计算无需任何高维重构，效率得以保留。

**4. 外部 logits 锚定防止代理坍缩。** 单靠梯度对齐容易让代理崩掉，IPROX 借鉴知识蒸馏，用前向 KL 把代理输出分布锚到目标（teacher）模型上：$\mathcal{L}_{KL}=\tau^2\frac{1}{|B|}\sum_z \mathrm{KL}(\mathrm{softmax}(f_\theta(z)/\tau)\,\|\,\mathrm{softmax}(f_{\theta'}(z)/\tau))$。最终阶段二目标是 $\min_{\theta'}\mathcal{L}_{GA}+\lambda_{KL}\mathcal{L}_{KL}$，KL 项提供稳定的对齐基底，梯度对齐项负责精修影响力一致性。

## 实验关键数据
**设置**：候选训练集用 DOLLY，评测用 TyDiQA（多语 QA）、MMLU（多选）、BBH（推理）。目标模型覆盖 Llama3.2-3B、Gemma3-4B、Qwen3-4B、Qwen2-7B 四个家族。目标模型先在 5% 数据上 warm-up，按影响力选 Top-5% 数据，全量微调 4 个 epoch；IPROX 只用 1% 数据源构造代理（其中 10% 当探针集）。$\rho$ 为压缩稀疏度（被移除参数比例）。

### 主实验表格（vs 现成代理，TracIn，节选 Avg.）

| 目标模型 | 代理 | #Params | MMLU | BBH | TyDiQA | Avg. |
|---|---|---|---|---|---|---|
| Qwen3-4B | 现成 Qwen3-1.7B | 1.7B | 69.65 | 74.44 | 47.35 | 63.81 |
| Qwen3-4B | **IPROX ρ=0.7** | **1.5B** | 69.94 | 74.62 | 47.98 | **64.18** |
| Qwen3-4B | IPROX ρ=0.3 | 3.1B | 70.15 | 75.18 | 50.63 | 65.32 |
| Llama3.2-3B | 现成 Llama3.2-1B | 1B | 55.89 | 47.31 | 38.84 | 47.35 |
| Llama3.2-3B | IPROX ρ=0.3 | 2.5B | 56.77 | 49.16 | 40.98 | 48.97 |
| Gemma3-4B | 现成 Gemma3-1B | 1B | 59.61 | 47.31 | 25.43 | 44.12 |
| Gemma3-4B | IPROX ρ=0.3 | 3B | 59.36 | 49.63 | 32.19 | 47.06 |

亮点：**Qwen3-4B 上，1.5B 的 IPROX（64.18）反超 1.7B 现成代理（63.81）**——更小却更强。某些情况下（Qwen3-4B ρ=0.3 的 BBH、Qwen2-7B ρ=0.3 的 TyDiQA）IPROX 选的数据甚至胜过用目标模型自己选的数据。

### 对比基线 / 效率表格

| 对比项 | Llama3.2-3B Avg. | Gemma3-4B Avg. |
|---|---|---|
| Layer Extraction | 46.16 | 45.00 |
| Influence Scorer | 45.70 | 44.76 |
| **IPROX (ρ=0.3)** | **48.97 (+2.81)** | **47.06 (+1.99)** |

| 计算开销 (Llama3.2-3B, 单卡 GH200) | Stage1 | Stage2 | 影响力计算 |
|---|---|---|---|
| Llama3.2-3B 全模型 | – | – | ~90 min |
| Llama3.2-1B 现成代理 | – | – | ~40 min |
| **IPROX (ρ=0.3~0.7)** | ~2 min | ~3–5 min | ~38–44 min |

FLOPs 上 ρ=0.7 相比 3B 全模型省下 140+ TFLOPs；代理构造（两阶段）只占不到 10 分钟额外开销，效率收益主要来自影响力计算变快。

### 关键发现
- **任务类型有讲究**：增益在 TyDiQA（开放域 QA，与 Dolly 分布近）上最明显，MMLU（复杂推理，分布远）增益有限——与命题 4.1 一致（分布偏移越大，误差界越松）。
- **更大代理更好**：四个家族都呈现代理越大性能越高的单调趋势，验证了"算力↔性能"可控权衡。
- **跨影响力估计器一致**：换成 Influence Function（K-FAC 实现），IPROX 仍在 BBH/TyDiQA 胜过现成代理、MMLU 持平，结论与 TracIn 一致。
- **为何有效**：低稀疏代理（ρ=0.3）的子空间亲和度（SA）更高（更贴目标任务方向），高稀疏代理（ρ=0.7）的最近邻距离（1-NND）更大（选出数据更多样）——IPROX 把选择导向任务相关方向、又借稀疏保留多样性。
- **探针集**：尺寸增大到 3× 后收益饱和、5× 报酬递减；探针多样性下降（注入冗余）会单调拉低性能。

## 亮点与洞察
- **范式转换**：把"选一个偏好假定的现成小模型"换成"从目标模型派生一个保留影响力的代理"，第一次给"代理与目标模型在影响力估计上对齐"提供了系统方法。
- **目标函数纠偏**：精准点出标准 SVD 的"重构误差最优"和"影响力保留"目标错位，并用二阶矩重加权 + K-FAC 把"保留影响力"变成一个可解的加权低秩近似，理论（命题 4.1）与方法严丝合缝。
- **效率与正确性兼顾的工程巧思**：低秩空间内对齐梯度（而非高维重构）+ 瘦 SVD 探针估计，使得"既省算力又对齐影响力"不再是 trade-off 两难。
- **"小胜大"的反直觉结论**：影响力保持的小代理能选出比大现成代理、甚至比目标模型自己更泛化的数据。

## 局限与展望
- **压缩有天花板**：embedding 层和 LM head 不宜压缩，且秩降到原尺寸约 10% 以下模型质量会骤降，意味着代理无法无限缩小，影响力保持在极端稀疏下难以完全恢复。
- **分布偏移敏感**：理论界与实验都表明，训练/验证分布差距大时（如 MMLU）增益有限，方法更适合训练数据与目标任务相近的场景。
- **依赖目标模型可访问**：要做 IPSVD 和梯度对齐需访问目标模型权重与梯度，纯黑盒目标模型不适用（这与"现成代理"路线的适用面不同）。
- 论文未给两阶段各自的逐项消融（如只 Stage1 / 只 KL）于正文主表，组件贡献的拆解主要在附录。

## 相关工作与启发
- **梯度数据选择**：TracIn（沿轨迹累加梯度相似度）、Influence Function（逆 Hessian 近似 LOO）是被加速的对象；IPROX 与"简化影响力计算本身"的工作（DataInf、LESS 的子集外推等）正交——它加速的是承载影响力计算的模型。
- **LLM 分解式压缩**：ASVD（含激活模式）、CALDERA（低秩+量化）、MoDeGPT（Nyström 近似整块）、ShortGPT（层重要性打分）——IPROX 把"压缩"从"保性能"目标改写为"保影响力"目标，是对这条线的重新定向。
- **启发**：当某个下游目标（这里是影响力/数据选择）与压缩的默认目标（重构误差）不一致时，应当把下游目标显式写进压缩的损失里——二阶矩重加权是一种通用且可解的实现路径，可能迁移到"为某特定任务定制压缩代理"的更多场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把"代理"从现成小模型改为"从目标模型派生的影响力保持代理"，并给出影响力加权 SVD + 低秩梯度对齐的完整方案，视角新且有理论支撑。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 4 个模型家族、2 类影响力估计器、3 个评测任务，含效率/探针/多样性多维分析；扣分在正文缺两阶段组件的直接消融、候选集偏单一（Dolly 为主）。
- **写作质量**: ⭐⭐⭐⭐ 动机—理论—方法—实验逻辑清晰，命题与加权目标推导衔接自然，图表支撑到位。
- **价值**: ⭐⭐⭐⭐ 让梯度数据选择对多十亿参数 LLM 更可扩展（影响力计算从 ~90min 降到 ~40min 且更准），对 SFT 数据治理有实际工程价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Difficulty–Diversity Collaborative Filtering for Data-Efficient LLM Fine-Tuning](difficultydiversity_collaborative_filtering_for_data-efficient_llm_fine-tuning.md)
- [\[ICLR 2026\] Neuron-Aware Data Selection in Instruction Tuning for Large Language Models](neuron-aware_data_selection_in_instruction_tuning_for_large_language_models.md)
- [\[ICLR 2026\] Explainable Token-level Noise Filtering for LLM Fine-tuning Datasets](explainable_token-level_noise_filtering_for_llm_fine-tuning_datasets.md)
- [\[ICLR 2026\] CPQS-Tuning: A Model Self-Perception-Based Data Filtering Algorithm for Efficient Instruction Fine-Tuning](cpqs-tuning_a_model_self-perception-based_data_filtering_algorithm_for_efficient.md)
- [\[ICLR 2026\] Mitigating Non-IID Drift in Zeroth-Order Federated LLM Fine-Tuning with Transferable Sparsity](mitigating_non-iid_drift_in_zeroth-order_federated_llm_fine-tuning_with_transfer.md)

</div>

<!-- RELATED:END -->
