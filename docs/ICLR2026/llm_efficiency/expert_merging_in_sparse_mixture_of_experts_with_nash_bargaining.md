---
title: >-
  [论文解读] Expert Merging in Sparse Mixture of Experts with Nash Bargaining
description: >-
  [ICLR 2026][LLM效率][Sparse MoE] 把稀疏 MoE 的"专家合并"重新解释为专家之间的合作—竞争博弈，用纳什议价解（Nash Bargaining Solution）从第一性原理推出每个专家的合并系数，并配上复数动量加速跨层传播，做出了 NAMEx 这套统一替换 CAMEx 启发式加权的合并框架。
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "Sparse MoE"
  - "Expert Merging"
  - "Nash Bargaining"
  - "博弈论"
  - "Complex Momentum"
  - "CAMEx"
---

# Expert Merging in Sparse Mixture of Experts with Nash Bargaining

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=JLe9xfd0ln](https://openreview.net/forum?id=JLe9xfd0ln)  
**代码**: [https://github.com/anh147/NAMEx](https://github.com/anh147/NAMEx)  
**领域**: LLM 高效化 / Sparse MoE / 专家合并  
**关键词**: Sparse MoE, Expert Merging, Nash Bargaining, 博弈论, Complex Momentum, CAMEx  

## 一句话总结
把稀疏 MoE 的"专家合并"重新解释为专家之间的合作—竞争博弈，用纳什议价解（Nash Bargaining Solution）从第一性原理推出每个专家的合并系数，并配上复数动量加速跨层传播，做出了 NAMEx 这套统一替换 CAMEx 启发式加权的合并框架。

## 研究背景与动机
**领域现状**：稀疏 MoE（SMoE）靠路由器为每个 token 选取少量专家，在保持算力的同时把模型容量做大。除了路由这条主线，还有一条被低估的方向——**专家合并（expert merging）**：不是为每个输入挑专家，而是把所有专家参数融合成一个统一模型，特别适合部署/显存受限、或自回归与跨域迁移场景。

**现有痛点**：主流合并方法（SMEAR 的软合并、top-k 聚合、以及更先进的 CAMEx 曲率感知合并）本质都是**启发式加权平均**——要么用路由权重、要么用自然梯度调整几何，但都缺一个**有原则的加权机制**来刻画"哪个专家该出多少力"。更具体地，CAMEx 的动态变体 EP-CAMEx 把一个 base expert 沿层传播以促进跨层通信，结果却**反而打不过它的静态版本**，作者归因为专家贡献之间缺乏协调。

**核心矛盾**：专家之间并非简单可加——它们既有**合作**（输出相似、互相增益）又有**竞争/对抗**（梯度方向冲突）的混合动态（论文 Figure 1 显示不同架构、不同层的专家余弦相似度模式差异巨大），而线性平均完全无视这种结构性博弈。

**本文目标**：给专家合并一个从第一性原理出发的系数推导，使融合既公平（Pareto 高效）又能区分合作与对抗。

**核心 idea**：**【博弈论视角】** 把每个专家的 domain-vector（相对 base expert 的偏移）当成多任务学习里的"任务梯度"，把专家合并建模成一场议价博弈，用纳什议价解求最优更新方向，再叠加复数动量解决 EP-CAMEx 收敛慢的问题。

## 方法详解

### 整体框架
NAMEx 接在 CAMEx 的专家传播框架上：每层先算出各专家相对 base expert $E_m$ 的 domain-vector $\tau_i = E_i - E_m$，把它们当作互相博弈的"任务梯度"，解一个纳什议价方程得到系数 $\alpha_i$，用 $\Delta E=\sum_i \alpha_i \tau_i$ 去更新 base expert 并跨层传播；再用复数动量缓冲累积这个更新方向以加速收敛。最后仍保留 CAMEx 的曲率感知项做输入相关的精修。

```mermaid
flowchart LR
    A["第 l 层专家<br/>E₁…E_N + base E_m"] --> B["domain-vector<br/>τᵢ = Eᵢ − E_m"]
    B --> C["纳什议价方程<br/>GᵀG·α = 1/α"]
    C --> D["更新方向<br/>ΔE = Σ αᵢτᵢ"]
    D --> E["复数动量累积<br/>μ = βμ + ΔE"]
    E --> F["传播 base expert<br/>E_m ← E_m + ℜ(γμ)"]
    F --> G["曲率感知精修<br/>+ η Σ Mᵢ(sᵢ∗τᵢ)"]
    G --> H["下一层"]
```

### 关键设计

**1. 把专家合并写成议价博弈（BEM Problem）：让系数自己"谈"出来。** 作者沿用 Navon 等人在多任务学习里的纳什议价框架，把每个专家设为一名玩家，其效用函数定义为 $u_i(\Delta E)=\tau_i^\top \Delta E$，议价的"不达成点"（disagreement point）设为 0（即不更新 base expert），可行集是半径 $\epsilon$ 的球 $B_\epsilon$。纳什解要求在 Pareto 高效（Axiom 3.1，没人能在不损害他人的前提下单独变好）约束下最大化各玩家相对不达成点收益的乘积，等价于求 $\arg\max_{\Delta E\in B_\epsilon}\sum_i \log(\Delta E^\top \tau_i)$。Lemma 3.2 给出闭式结构：最优方向 $\Delta E^*=\sum_i \alpha_i \tau_i$，其中系数向量满足
$$G^\top G\,\alpha = 1/\alpha,$$
$G=[\tau_1,\dots,\tau_N]$ 是 domain-vector 拼成的矩阵，$1/\alpha$ 是逐元素倒数。这条方程正是 NAMEx 与所有启发式加权的分水岭——系数不是人手指定，而是博弈均衡的产物。

**2. 系数 α 如何编码合作与对抗：一条可解读的更新律。** 把 Lemma 3.2 展开到单个专家，可得
$$\alpha_j\|\tau_j\|^2 + \sum_{i\neq j}\alpha_i\tau_i^\top\tau_j = \frac{1}{\alpha_j}.$$
其中 $\sum_{i\neq j}\alpha_i\tau_i^\top\tau_j$ 正是第 $j$ 个专家与其余专家的交互项：若它为正，说明其他专家在**协助** $j$（方向一致、合作），此时 $\alpha_j$ 自动**变小**（不用自己太用力）；若为负，说明其他专家在**对抗/拖累** $j$，$\alpha_j$ 就**变大**以维持等式成立、保住自己的贡献。特别地，当所有 $\tau_j$ 正交时退化为 $\alpha_j=1/\|\tau_j\|$ 的尺度不变解；当各 domain-vector 范数近似相等时，EP-CAMEx 恰好是忽略专家间交互的"平凡解"——这就从理论上解释了为什么 NAMEx 能严格泛化并超过 CAMEx 系方法。

**3. 复数动量加速跨层传播：补上 EP-CAMEx 收敛慢的短板。** EP-CAMEx 的 base expert 只能更新"模型层数"那么多步，后期收敛不充分，这正是它打不过静态 CAMEx 的根因。NAMEx 引入 Lorraine 等人的**复数动量**（complex momentum，已被证明在合作—对抗博弈中比一阶方法更稳更快），维护一个复数缓冲 $\mu^{(j)}\in\mathbb{C}^d$：
$$\mu^{(j+1)}=\beta\mu^{(j)}+\Delta E^{(j)},\qquad E_m^{(j+1)}=E_m^{(j)}+\Re(\gamma\mu^{(j+1)}),$$
$\beta\in\mathbb{C}$ 是复动量系数，$\Re(\cdot)$ 取实部。作者给出基于谱半径的收敛速率界（Proposition 3.5 / Theorem C.3），证明存在 $\gamma,\beta$ 使更新收敛；实验里 $\beta$ 的辐角 $\phi\neq 0$（即真正用上复数而非退化为实动量）是性能关键。

**4. 两种预算分配：NAMEx 与 NAMEx-Full。** 为对齐 EP-CAMEx 的训练时间，议价预算固定为每 batch 20 次迭代。**NAMEx** 只在第一层算一次 $\alpha$ 然后逐层复用；**NAMEx-Full** 把预算均摊到每一层、逐层重解纳什方程（受 Navon 启发），后者更贴合"专家交互逐层变化"的观察，因而在多数任务上最强。

## 实验关键数据

### 主实验：语言建模 / 文本分类 / 图像分类
- **WikiText-103 语言建模**（Table 1，越低越好）：NAMEx-Full-Mom 在 small/medium 两个尺度都拿到最低困惑度。Medium 尺度 Test PPL 从 SMoE(Top-2) 的 35.55、CAMEx 的 36.53 降到 **35.37**；NAMEx 系普遍优于对应的 CAMEx/EP-CAMEx 基线。
- **GLUE 文本分类**（Table 2，T5-Base，8 专家/层）：NAMEx-Full-Mom 在 7 个任务上**全部最优**。例如 SST-2 95.06（vs CAMEx 93.80）、MRPC 93.27、CoLA 60.13、RTE 78.15（vs EP-CAMEx 75.81）。

| 方法 | SST-2 | MRPC | CoLA | RTE | MNLI |
|------|------|------|------|-----|------|
| SMoE (Top-2) | 94.35 | 91.04 | 58.43 | 74.98 | 86.72 |
| CAMEx | 93.80 | 91.16 | 58.57 | 74.72 | 86.44 |
| EP-CAMEx | 93.69 | 91.01 | 58.29 | 75.81 | 86.94 |
| NAMEx | 94.46 | 92.01 | 58.81 | 75.09 | 86.96 |
| NAMEx-Full | 94.82 | 92.80 | 59.63 | 77.83 | 87.23 |
| **NAMEx-Full-Mom** | **95.06** | **93.27** | **60.13** | **78.15** | **87.45** |

- **ImageNet-1k 图像分类与鲁棒性**（Table 3，Swin-MoE）：NAMEx-Full-Mom Acc@1 **84.52**（vs CAMEx 83.29、SMoE 83.15）。分布偏移下优势更明显——ImageNet-A 上 NAMEx-Mom/Full-Mom 把准确率从 CAMEx 的 25.45 拉到 **35.05/35.27**，复数动量在 corruption 场景增益最大。

### 大模型规模实验
- 集成进 **DeepSeek-MoE (16B)** 与 **Qwen1.5-MoE (14B)**，在 MMLU/GSM8K/ARC 上、跨 Linear/Cosine/Stable-MoE 三种路由策略、zero-shot 与 SmolTalk 微调两种设定下，NAMEx-Full 一致超过基线与 EP-CAMEx（如 Stable-MoE 路由微调后 MMLU 46.42 vs base 46.17、ARC 50.64 vs 50.28），证明可扩展性。

### 消融实验

| 消融维度 | 设置 | 发现 |
|---------|------|------|
| 复数动量辐角 $\phi$ (Table 5) | $\phi\in\{\pm\pi/6,\pm\pi/12,0\}$ | $\phi=0$（实动量）所有任务下滑，**非零辐角是关键**；最优 $\phi$ 需按任务调 |
| 更新步频 $\Delta l$ (Table 6) | $\Delta l\in\{1,2,5,L\}$ | 更频繁地重解 $\alpha$（$\Delta l$ 小）通常更准，但 runtime 从 0.69s 涨到 4.70s，准确率—效率权衡 |
| 不达成点 (Table 4) | 0 vs mean | 两者差异极小，方法对 disagreement point 选择不敏感 |

### 关键发现
1. **博弈结构本身就有用**：即便不加动量，NAMEx-Full 在 clean benchmark 上已能追平 NAMEx-Mom，说明逐层纳什解的价值。
2. **复数动量在对抗/分布偏移场景增益最大**（ImageNet-A 提升十个点级别）。
3. 合成三专家实验（Figure 11）显示平均合并可能落不到 Pareto 集，而 NAMEx 倾向产出 Pareto 高效解，从几何上印证"不被 EP-CAMEx 或线性平均支配"。

## 亮点与洞察
- **视角换得漂亮**：把"专家合并系数怎么定"这个一直靠启发式糊弄的问题，干净地映射到成熟的纳什议价/多任务学习框架，系数有了第一性原理来源。
- **可解读性强**：交互项 $\sum_{i\neq j}\alpha_i\tau_i^\top\tau_j$ 的正负直接对应合作/对抗，并自动调节 $\alpha_j$ 大小，这种"会谈判的合并"比黑箱权重更有说服力。
- **理论与诊断统一**：用 Lemma 3.2 证明 EP-CAMEx 只是忽略交互的平凡解，既解释了"EP-CAMEx 为何反不如静态 CAMEx"，又给出了改进方向，逻辑闭环。
- **即插即用**：能套在 Swin-MoE、ACMoE、T5-MoE、DeepSeek-MoE、Qwen-MoE 等多种架构上，且兼容不同路由策略。

## 局限与展望
- **额外计算开销**：每层解 $G^\top G\alpha=1/\alpha$ 并多次迭代，更频繁更新（$\Delta l$ 小）会显著拉高 runtime，存在准确率—效率权衡；大模型上需固定预算来对齐训练时间。
- **超参敏感**：复数动量辐角 $\phi$ 需逐任务调优，缺乏自动选取机制。
- **增益幅度偏温和**：在大模型 zero-shot/微调上的提升多为零点几个百分点，是否值得额外复杂度需结合场景权衡。
- **展望**：作者明确指出**四元数动量（quaternion momentum）**是有前景的下一步，可进一步丰富专家传播的动态结构。

## 相关工作与启发
- **CAMEx / EP-CAMEx**（Nguyen et al., 2025）：本文的直接前身，提供曲率感知合并与跨层传播框架；NAMEx 把它的传播步替换为纳什解并证明前者是平凡特例。
- **多任务学习的纳什议价**（Navon et al., 2022, Nash-MTL）：方法论母体，本文把"任务"换成"专家"完成迁移。
- **复数动量优化**（Lorraine et al., 2022）：在合作—对抗博弈中稳健加速，被借来解 EP-CAMEx 收敛慢。
- **软合并 / top-k 聚合**（SMEAR, Lory 等）：被统一为 CAMEx/NAMEx 框架下的特例，凸显启发式加权的局限。
- **启发**：把"参数融合该怎么加权"理解为多智能体博弈，是一个可迁移到模型融合（model merging）、LoRA 合并、联邦聚合等更广场景的思路——凡是"多个方向向量要融成一个"的地方，纳什议价都可能比平均更有原则。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把纳什议价/博弈论引入 MoE 专家合并是少见且自洽的视角，并用理论证明前作是其平凡特例。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖语言/文本/图像三大模态 + 鲁棒性 + 16B/14B 大模型 + 多路由策略 + 充分消融，五随机种子，规模到位。
- **写作质量**: ⭐⭐⭐⭐ 动机—理论—算法—实验逻辑清晰，公式与可解读性叙述结合得好；理论部分细节较密，需一定背景。
- **价值**: ⭐⭐⭐⭐ 给专家合并提供了有原则的加权范式，即插即用且可迁移到更广的模型融合问题，温和但稳定的增益具实用意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)
- [\[ICLR 2026\] Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models](not_all_models_suit_expert_offloading_on_local_routing_consistency_of_mixture-of.md)
- [\[ICLR 2026\] DirMoE: Dirichlet-Routed Mixture of Experts](dirmoe_dirichlet-routed_mixture_of_experts.md)
- [\[ICLR 2026\] Understanding the Mixture-of-Experts with Nadaraya-Watson Kernel](understanding_the_mixture-of-experts_with_nadaraya-watson_kernel.md)
- [\[ACL 2026\] Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference](../../ACL2026/llm_efficiency/alloc-moe_budget-aware_expert_activation_allocation_for_efficient_mixture-of-exp.md)

</div>

<!-- RELATED:END -->
