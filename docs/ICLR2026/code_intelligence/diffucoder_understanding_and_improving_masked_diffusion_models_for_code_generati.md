---
title: >-
  [论文解读] DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation
description: >-
  [ICLR 2026][代码智能][掩码扩散模型] 本文训练了一个 7B 的掩码扩散代码模型 DiffuCoder，提出局部/全局 AR-ness 指标系统刻画扩散 LLM 的"非自回归"解码行为，并设计 coupled-GRPO（互补掩码耦合采样的扩散原生 RL 方法），在 EvalPlus 上提升 4.4%。
tags:
  - "ICLR 2026"
  - "代码智能"
  - "掩码扩散模型"
  - "dLLM"
  - "代码生成"
  - "GRPO"
  - "耦合采样"
---

# DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=58NA3unZj5](https://openreview.net/forum?id=58NA3unZj5)  
**代码**: https://github.com/apple/ml-diffucoder (有)  
**领域**: 代码生成 / 扩散语言模型 / 强化学习  
**关键词**: 掩码扩散模型, dLLM, 代码生成, GRPO, 耦合采样

## 一句话总结
本文训练了一个 7B 的掩码扩散代码模型 DiffuCoder，提出局部/全局 AR-ness 指标系统刻画扩散 LLM 的"非自回归"解码行为，并设计 coupled-GRPO（互补掩码耦合采样的扩散原生 RL 方法），在 EvalPlus 上提升 4.4%。

## 研究背景与动机

**领域现状**：当前主流的代码大模型几乎都建立在自回归（AR）范式上（Qwen2.5-Coder、OpenCoder），从左到右逐 token 生成。近期掩码扩散模型（MDM）被扩展成扩散 LLM（dLLM，如 LLaDA、Dream），它们对整段序列并行去噪、可全局规划内容，性能已能与同规模 AR 模型持平。直觉上代码生成天然契合扩散范式——写代码本身就是来回跳改、非顺序的过程。

**现有痛点**：开源 dLLM 在代码任务上到底表现如何、其训练与推理机制如何解释，仍是黑箱。已有的 dLLM 后训练（LLaDA1.5 用 DPO，d1、MMaDA 用 GRPO）要么收益微弱，要么严重依赖 semi-AR 解码（即 block decoding，用很小的 block 把序列切块逐块生成）。而 semi-AR 解码恰恰把因果偏置重新塞回生成过程，背离了扩散"全局规划"的本质。

**核心矛盾**：一方面我们想用扩散的非自回归特性（并行、全局规划）；另一方面现有 dLLM 究竟有多"非自回归"无人量化，且把 GRPO 套到扩散上时，token 似然只能靠蒙特卡洛采样估计——采样次数有限就方差大、效率低，d1 为此干脆全掩码单次前向，但这样估计是有偏的。

**本文目标**：(1) 量化并理解 dLLM 的解码行为相比 AR 有何不同；(2) 设计一个不依赖 semi-AR、尊重扩散非自回归本质的 RL 后训练方法。

**切入角度**：作者先造一个强 dLLM 测试床（DiffuCoder，130B code token 训练），再用自定义的 AR-ness 指标拆解它的解码模式，发现"采样温度不仅多样化 token 选择、还多样化生成顺序"，这为 RL rollout 提供了丰富搜索空间；顺着这个观察设计耦合采样降低似然估计方差。

**核心 idea**：用一对互补掩码（complementary mask）做耦合采样，在保证全 token 覆盖的前提下无偏且低方差地估计 token 对数似然，从而把 GRPO 改造成扩散原生的 coupled-GRPO。

## 方法详解

### 整体框架

DiffuCoder 的工作分三段：先把一个 AR 代码模型适配训练成 7B 扩散代码模型（四阶段管线），再用新提出的 AR-ness 指标剖析它的解码行为（理解阶段），最后基于剖析结论设计 coupled-GRPO 做强化学习后训练。输入是 Qwen2.5-Coder 基座 + 大规模代码语料，输出是经 RL 强化、解码更并行的 DiffuCoder-Instruct。

掩码扩散的基本设定：前向过程把 $x_0$ 逐步腐蚀为带 [MASK] 的噪声序列，反向过程学一个去噪器 $f_\theta$ 重建被掩码的 token，训练目标是 ELBO 导出的加权交叉熵 $L_t = \frac{1}{t}\mathbb{E}_q[-\sum_n \delta_{x_t^n,m}\,(x_0^n)^\top \log f_\theta(x_t)^n]$，只在被掩码位置计损。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Qwen2.5-Coder 基座<br/>+ 130B 代码语料"] --> B["四阶段训练管线<br/>适配预训练→中训练→指令微调"]
    B --> C["局部 / 全局 AR-ness 指标<br/>量化解码有多自回归"]
    C -->|发现高温多样化生成顺序<br/>pass@k 有提升空间| D["coupled-GRPO 耦合采样<br/>互补掩码对降方差估似然"]
    D --> E["DiffuCoder-Instruct<br/>EvalPlus +4.4%、解码更并行"]
```

### 关键设计

**1. 四阶段训练管线：把 AR 代码基座适配成扩散代码模型**

要做扩散原生的 RL，先得有一个够强的扩散基座。作者不从零训，而是从 Qwen2.5-Coder 适配（continual pre-training），用 Gong et al. 的 adaptation 方法把 AR 模型改造成掩码扩散模型。整条管线分四阶段：**Stage 1 适配预训练**（RefineCode + Stackv2 代码语料，packed、无条件）；**Stage 2 中训练**（mid-training，用 OpenCoder 退火数据做承上启下的 annealing 阶段，跑 4 epoch 共 65B token）；**Stage 3 指令微调**（OpenCoder 的 436K SFT 样本，padded、条件式，增强指令跟随）；**Stage 4 coupled-GRPO** 后训练（见设计 3）。

一个关键的工程经验是数据质量比数据量更重要：Stage 1 用 700B token 训练反而比只用 65B 在下游验证集上更差，作者推测 continual pre-training 对数据质量很敏感，于是对 Stage 1 早停、只用 65B token 作为 Stage 2 的起点。最终 DiffuCoder 在 130B code token（Stage 1+2）后，base 性能已与 Qwen2.5-Coder、OpenCoder 持平——但所有 dLLM 在指令微调后只有微弱提升，远不及 AR 模型同样数据 SFT 的增益，这个"指令微调收益鸿沟"正是促使作者转向 RL 的导火索。

**2. 局部/全局 AR-ness 指标：量化 dLLM 解码到底有多"自回归"**

要利用非自回归的好处，得先知道当前 dLLM 实际有多非自回归。作者定义两个指标。**局部 AR-ness@k（next-token 模式）**：统计解码步中，新揭开的 token 与其前 $k$ 个已生成 token 是否构成严格连续递增序列的比例——衡量"是否在做下一 token 预测"，$k$ 越大越难维持，值越高越像 AR。**全局 AR-ness@k（earliest-mask 模式）**：统计每步揭开的 token 是否落在所有剩余掩码位置中最靠左的 $k$ 个之内的平均比例——衡量"是否总在填最左的空"，即从左到右的填充倾向。纯 AR 解码两个指标恒等于 1。

基于这两个指标，作者得出几个有画面感的结论：(1) dLLM 解码确实没那么 AR——相当一部分 token 既不是从最左掩码、也不是从下一位恢复，但两指标都更接近 1 而非 0，说明文本本身就带 AR 结构，扩散模型无论从零训还是从 AR 适配都自然捕获了它；(2) **entropy sink（熵汇）现象**——首个前向步每个位置的置信度呈"L"形：紧贴前缀右侧的位置拿到更强的位置信号和更近的上下文，被赋予不成比例的高置信度，这正是 AR-ness 的来源；(3) 代码相比数学有更低的全局 AR-ness 均值、更高方差——模型生成代码时倾向先产后面的 token、把一些靠前的空留到很晚，就像程序员在代码里来回跳改；(4) **温度的双重作用**——AR 模型里温度只影响 token 选择，dLLM 里温度同时影响 token 选择和生成顺序，把温度从 0.2 提到 1.0~1.2 会显著降低 AR-ness 并大幅抬高 pass@k，暴露出可被 RL "激发出来"的潜在能力。

**3. coupled-GRPO 耦合采样：互补掩码对实现低方差、全覆盖的似然估计**

把扩散过程视作 MDP（状态 $s_t=(c,t,x_t)$、动作 $a_t=x_{t-1}$），就能套 GRPO，但卡点在 token 对数似然的蒙特卡洛估计：$L_t$ 只对被掩码位置计损，采样次数有限时既低效又高方差。基线 d1 干脆把所有 completion token 全掩码、单次前向（等价于在 $t=T$ 采一次），但由于 entropy sink，高熵 token 集中在左侧，RL 会过度更新靠前的 token，估计是有偏的；d1 还随机掩 15% 条件 token，在代码任务上反而让 completion 似然估计不可靠（代码比数学更要求 token 级精度）。

coupled-GRPO 的做法是**耦合采样**：取 $\lambda$ 对满足 $t+\hat{t}=T$ 的时间步对，对同一 completion 采两个**互补掩码**——每个掩码各遮一部分 token，两者并起来恰好覆盖全部 token，即每个 token 在两次前向中恰好被揭开一次。对数似然估计为
$$\log \pi_\theta(o_i^k \mid c, o_{i,t<T}^k) = \frac{1}{\lambda+1}\Big[\sum_{t+\hat{t}=T}\big(L_t(x_t)+L_{\hat t}(x_{\hat t})\big)+L_T(x_T)\Big]_i^k,\quad \delta_{x_t,m}+\delta_{x_{\hat t},m}=1.$$
这样保证 (1) 每个 token 的对数似然至少被算一次、都拿到非零学习信号；(2) 每个 token 是在真实的部分掩码上下文（而非永远全掩码）下被评估，估计更准，且比基线多 $2\lambda$ 个样本。实践取 $\lambda=1$。作者从 antithetic variates（对偶变量）角度证明了它的方差缩减性质。优势用组内相对优势 $A_i = r(o_i)-\frac{1}{G}\sum_j r(o_j)$，并可选 leave-one-out 做无偏估计；奖励由代码格式奖励 + 测试用例执行通过率（正确性奖励）组成。

### 损失函数 / 训练策略
RL 阶段从 Acecoder-87K 里挑 21K 带可验证测试用例的难样本，rollout 温度设为 1.2（对应 pass@10 更高的区间），基于 Open-R1 codebase 在 8~10 个节点（每节点 8×H100）上训练。GRPO 目标采用 PPO 式裁剪 + KL 惩罚（式 3/4）。一个有趣现象：RL 微调把评测的最优采样温度从 0.2 推到 0.3~0.4，说明训练锐化了每 token 分布。

## 实验关键数据

### 主实验

7/8B 规模代码基准对比（EvalPlus = HE+ 与 MBPP+ 均值；±为指令模型相对 base 的绝对变化）：

| 模型 | HumanEval+ | MBPP+ | EvalPlus | BigCodeBench(C) Full |
|------|-----------|-------|----------|----------------------|
| Qwen2.5-Coder (base) | 51.8 | 61.4 | 56.6 | 46.1 |
| DiffuCoder (base) | 60.4 | 60.9 | 60.6 | 40.2 |
| Dream-Instruct | 53.7 | 56.1 | 54.9 | 10.6 |
| DiffuCoder-Instruct | 65.2 | 61.9 | 63.6 | 35.7 |
| **+ coupled-GRPO** | **68.3** (+7.9) | **67.5** (+6.6) | **67.9** (+7.3) | 40.4 |

DiffuCoder base 经 130B code token 训练即与 Qwen2.5-Coder/OpenCoder 持平；coupled-GRPO 仅用 21K 样本就把 EvalPlus 拉高 4.4%（相对 Instruct）。

### 消融实验

GRPO 后训练各变体（HumanEval / MBPP / BigCodeBench，取温度集 {0.2,0.3,0.4} 最优）：

| 配置 | HumanEval+ | MBPP+ | 说明 |
|------|-----------|-------|------|
| DiffuCoder-Instruct | 65.2 | 61.9 | 起点 |
| + coupled-GRPO | 68.3 (+3.1) | 67.5 (+5.6) | 完整方法 |
| + coupled-GRPO (LOO) | 62.2 (−3.0) | 68.5 (+6.6) | 留一无偏优势 |
| w/ full mask completion | 59.1 (−6.1) | 65.1 (+3.2) | 退化成 d1 式全掩码 |
| w/ decoupled sampling | 62.8 (−2.4) | 66.4 (+4.5) | 同样本数但无互补约束 |

### 关键发现
- **耦合采样是关键**：全掩码（d1 式）和去耦采样（同样本数、随机掩码）在 reward 曲线上都不稳定，HumanEval+ 比完整方法掉 6~9 分，证明"互补约束"而非"多采几次"才是收益来源。
- **rollout 温度敏感**：温度 1.2 优于 1.0，与 pass@10 趋势一致；高温提供更高的 rollout 多样性，RL 才有可强化的空间。
- **AR-ness 下降换来并行加速**：coupled-GRPO 后全局 AR-ness 降低，把解码步数砍半（≈2× 加速）时性能下降比训练前更小，说明并行度提升。
- **最优温度被锐化**：RL 后评测最优温度从 0.2 移到 0.3~0.4，与近期 AR LLM 的 RL 结论一致，提示这些方法可能可迁移到 dLLM。

## 亮点与洞察
- **AR-ness 指标把"非自回归到底多非"变得可测**：局部/全局两个角度配合 entropy sink 分析，第一次系统刻画了 dLLM 的解码序，这套度量本身就可复用到任何掩码扩散模型的行为分析。
- **温度在 dLLM 里是"双旋钮"**：同时调 token 选择和生成顺序——这个观察直接解释了为什么高温能撑大 pass@k 搜索空间，是把 RL 接到扩散上的认知前提。
- **互补掩码 = 对偶变量降方差**：用一对 $t+\hat{t}=T$ 的互补掩码同时拿到全覆盖与方差缩减，是把经典蒙特卡洛技巧（antithetic variates）干净地搬到 dLLM 似然估计上的巧思，可迁移到任何需要 ELBO 式 token 似然的扩散后训练。
- **不依赖 semi-AR**：相比 d1/MMaDA，coupled-GRPO 全程尊重扩散的全局并行本质，证明扩散原生 RL 是可行且有效的。

## 局限与展望
- 实验集中在 Python 代码三大基准（HumanEval/MBPP/BigCodeBench），未覆盖多语言、长程仓库级或复杂推理任务，"代码更适合扩散"的论断主要靠全局 AR-ness 间接支撑。
- coupled-GRPO 实践只验证了 $\lambda=1$，更大 $\lambda$ 的方差-成本权衡只在附录讨论；耦合采样仍需 2 次前向，相对全掩码基线有额外开销。
- entropy sink 的解释是假设性的（位置信号更强导致高置信度），属于现象级洞察，尚无机制层证明。
- BigCodeBench-Hard 等更难子集上 coupled-GRPO 偶有掉点，说明 RL 增益在高难度任务上不稳定。

## 相关工作与启发
- **vs d1 / MMaDA**：它们也用 GRPO 优化 dLLM，但严重依赖 block/semi-AR 解码、且用全掩码单次前向估似然（有偏，过度更新左侧 token）。本文用互补掩码耦合采样做无偏低方差估计，全程不依赖 semi-AR，故能进一步提升已指令微调的模型。
- **vs LLaDA / Dream**：同为开源 dLLM，但本文专攻代码、且给出可量化的解码行为分析（AR-ness、entropy sink），并配套扩散原生 RL，而非仅停留在"性能与 AR 持平"。
- **vs VRPO / DDPO / DPPO**：DDPO/DPPO 把连续扩散视作 MDP 做策略优化，VRPO 给 dLLM 引入高效 DPO 采样；本文沿用 MDP 视角但聚焦 GRPO 的似然估计方差问题，给出对偶变量式解法。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ AR-ness 指标 + 互补掩码耦合采样两点都是扩散代码 RL 的原创贡献
- 实验充分度: ⭐⭐⭐⭐ 解码行为分析详尽、消融到位，但基准偏 Python 单语言、$\lambda$ 选择验证有限
- 写作质量: ⭐⭐⭐⭐⭐ 从"理解解码行为"到"设计 RL"逻辑链清晰，图文配合好
- 价值: ⭐⭐⭐⭐⭐ 开源 7B 扩散代码模型 + 扩散原生 RL 框架，为 dLLM 后训练打下基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](../../ACL2026/code_intelligence/coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ICML 2026\] Locally Coherent Parallel Decoding in Diffusion Language Models](../../ICML2026/code_intelligence/locally_coherent_parallel_decoding_in_diffusion_language_models.md)
- [\[ICLR 2026\] Improving Code Localization with Repository Memory](improving_code_localization_with_repository_memory.md)
- [\[AAAI 2026\] Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning](../../AAAI2026/code_intelligence/towards_better_code_understanding_in_decoder-only_large_language_models_via_hie.md)
- [\[AAAI 2026\] DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](../../AAAI2026/code_intelligence/diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)

</div>

<!-- RELATED:END -->
