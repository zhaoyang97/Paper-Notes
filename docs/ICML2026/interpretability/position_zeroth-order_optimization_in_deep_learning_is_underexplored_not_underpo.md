---
title: >-
  [论文解读] Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered
description: >-
  [ICML2026 Spotlight][可解释性][零阶优化] 这是一篇 position paper，作者主张深度学习中的零阶（ZO）优化并非"无能"而是"未被充分探索"——他们沿算法/系统/评估三条主线给出 6 个论断（P1–P6），核心立场是：跳出"全空间逐元素估计器"的窠臼，转向子空间/谱域估计、前向单向流的系统级红利以及去混淆的评测协议，ZO 才能从内存高效微调的小众工具走向可扩展的训练范式。
tags:
  - "ICML2026 Spotlight"
  - "可解释性"
  - "零阶优化"
  - "方差控制"
  - "子空间优化"
  - "分布式训练"
  - "前向梯度"
---

# Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered

**会议**: ICML2026 Spotlight  
**arXiv**: [2605.15622](https://arxiv.org/abs/2605.15622)  
**代码**: 待确认  
**领域**: 优化  
**关键词**: 零阶优化, 方差控制, 子空间优化, 分布式训练, 前向梯度

## 一句话总结
这是一篇 position paper，作者主张深度学习中的零阶（ZO）优化并非"无能"而是"未被充分探索"——他们沿算法/系统/评估三条主线给出 6 个论断（P1–P6），核心立场是：跳出"全空间逐元素估计器"的窠臼，转向子空间/谱域估计、前向单向流的系统级红利以及去混淆的评测协议，ZO 才能从内存高效微调的小众工具走向可扩展的训练范式。

## 研究背景与动机

**领域现状**：零阶优化用有限差分 $\hat{\nabla}_{\mathbf{x}} f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{u}) - f(\mathbf{x})}{\mu}\mathbf{u}$ 估计梯度，绕过反向传播（BP）。2023 年 MeZO 把 ZO 从对抗样本/Prompt 调优这类输入级低维场景拉到大模型权重级微调，立刻引爆了"用前向 pass 省显存"的研究热潮（图 1 左：arXiv ZO 论文 2023 年后近乎指数增长）。

**现有痛点**：然而社区里同时弥漫着一种悲观主张——ZO 估计器方差随参数维度 $d$ 线性爆炸、查询代价不友好，因此"根本不可能 scale"。这种判断让 ZO 被默认锁在 LLM 微调这一类有强 task alignment 兜底的舒适区，无法承担从零训练或大规模 black-box 任务。

**核心矛盾**：作者认为这是"误诊"。多数所谓的 ZO 局限性，并非来自梯度无关学习的本质缺陷，而是来自三类"近视的工程实践"——(i) 把所有功夫都堆在估计器本身（estimator-centric）；(ii) 永远在原始全参数空间（full-space）操作；(iii) 永远在逐元素扰动（element-wise）形式下评估方差。这些选择把 ZO 的真正优势（前向单向、可分解为标量+随机种子、天然并行）全部掩盖掉了。

**本文目标**：把 ZO 现有研究地图重绘成"算法–系统–评测"三栈，把 6 个被忽略的关键点（P1–P6）立成靶子，让社区跳出梯度估计器的内卷，转向系统级、子空间级、评测级的红利挖掘。

**切入角度**：作者并非提出新的 ZO 算法，而是用一种诊断式视角——把 RGE 公式拆成方差/查询/方向导数三个分析维度（P1–P3），再把"子空间+谱"、"通信效率+管线并行"、"任务对齐的混淆效应"三个被低估的方向各立一个 position（P4–P6），最后落到 5 条具体的"call to action"。

**核心 idea**：ZO is underexplored, not underpowered——把 ZO 从"BP 的廉价替身"重新定义为"前向推理友好、天然分布式、可在子空间运行"的独立优化范式。

## 方法详解

这是一篇 position paper，没有新算法，"方法"指的是 6 个论断（P1–P6）背后那条共用的论证链。全文只借一个数学骨架——随机梯度估计器 RGE 的有限差分公式——把每个 position 都翻译成对它某个变量的改写或重新诠释：换 $\mathbf{u}$ 的分布、调 $m,n$ 的批量、取 $\mu \to 0$ 的极限、或用 $\mathbf{Pu}$ 替掉 $\mathbf{u}$。所有看似独立的主张因此都长在同一棵公式树上，论文唯一引用的训练形式也就是把 RGE / S-RGE / CGE / 前向梯度都塞进同一个 SGD 步 $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \hat{\nabla}_{\mathbf{x}} f(\mathbf{x}_t)$。

### 整体框架

6 条 position 分成前后两半。P1–P3 先划清"估计器中心"这套老范式的可行性边界：从方差控制（P1）走到方差–查询的权衡（P2），再把方向导数视角立成绕不开的基线（P3）。P4–P6 则把镜头从估计器本身拉远，去看三块被冷落的红利：子空间与谱域优化（P4）、前向单向流带来的系统级好处（P5）、以及评测里必须剥掉 task alignment 这层"混淆器"（P6）。最后 §4 把六条凝成 5 条 call to action（A1 评测协议、A2 跳出全空间、A3 生成式估计器、A4 ZO-native 系统栈、A5 拓宽应用前沿，尤其是量子计算与推理引擎栈复用）。

"underexplored"这个判断在表 1 里被量化：作者把 ICML'25 / NeurIPS'25 / ICLR'26 上 10 篇代表性 ZO 工作按 P1/P2/P3 三栏逐一打钩，结果几乎所有人都只满足 P1，没人同时管 P2 的查询代价和 P3 的前向梯度基线——社区的注意力被严重压在了一个角落。

### 关键设计

**1. 从 RGE 到子空间 RGE：把方差和维度脱钩**

针对的死结是 P1 与 P2：原始 ZO 估计器 $\hat{\nabla}_{\mathbf{x}}f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{u})-f(\mathbf{x})}{\mu}\mathbf{u}$（$\mathbf{u} \in \mathbb{R}^d$）的方差与参数维度 $d$ 同阶，而 mini-batch 平均压方差又很快遇到 diminishing returns，全空间里怎么都廉价不下来。作者的反应是干脆换一个空间——把扰动塞进一个低维子空间，写成 S-RGE：$\hat{\nabla}_{\mathbf{x}}f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{Pu})-f(\mathbf{x})}{\mu}\mathbf{Pu}$，其中 $\mathbf{P} \in \mathbb{R}^{d \times r}$、$r \ll d$、$\mathbf{u} \in \mathbb{R}^r$。

这个改写之所以站得住，是因为它有干净的几何解释。在方向导数极限 $\mu \to 0$ 下 $\mathbb{E}_{\mathbf{u}}[\hat{\nabla}_{\mathbf{x}}f] = \mathbf{PP}^\top \nabla_{\mathbf{x}} f$，当 $\mathbf{P}$ 列正交时 $\mathbf{PP}^\top$ 恰是把 FO 梯度投到 $\mathbf{P}$ 所张子空间的投影算子——S-RGE 就是 FO 梯度的"子空间近似"，方差顺势从 $O(d)$ 掉到 $O(r)$。$\mathbf{P}$ 只需对高斯矩阵做一次 QR 分解随机生成，而且可以"惰性更新"（隔若干步才刷一次），开销几乎可忽略。只要模型梯度本身近似低秩（Zhao et al. 2024 等已观察到这一现象），近似带来的精度损失远小于省下的方差。P4 进一步指出，这条路天然接得上谱域优化（如 Muon 的梯度正交化），把"低秩梯度结构"当成可利用的先验。

**2. 前向单向加共享种子，让分布式 ZO 只传一个标量**

这条把 ZO"必须靠随机扰动"这个看似的缺点，翻译成了系统优势。关键观察是本地 S-RGE 可以拆成 $\hat{\nabla}_{\mathbf{x}}f_i(\mathbf{x}) = \Delta_i \cdot \mathbf{Pu}_i$，其中 $\Delta_i = \frac{f_i(\mathbf{x}+\mu\mathbf{Pu}_i) - f_i(\mathbf{x})}{\mu}$ 只是个标量。于是 worker $i$ 不必回传整条梯度向量，只发标量 $\Delta_i$ 加上生成 $\mathbf{u}_i$ 的随机种子；中心节点用同一个种子在本地重建 $\mathbf{u}_i$（投影矩阵 $\mathbf{P}$ 也靠共享种子重建）再聚合，通信带宽就从 $O(d)$ 砍到了 $O(1)$。

红利不止在节点之间。单机内部用结构化扰动（按层/块/坐标）天然允许 feature reuse——只有被扰动那部分的激活会变，前向 pass 可以从扰动层切入而不必从输入重算（FZOO 已验证）。更深一层：ZO 把"前向算梯度"和"梯度立即可用"打包进了同一个前向 pass，pipeline 并行里 FO 训练特有的 1F1B 气泡（forward/backward 强耦合造成的）被直接消掉，pipeline 可以做成单向、近零气泡的"推理化"调度。同一个标量×高斯向量的结构还顺带给出一个隐私解释：ZO 估计天生带噪，能直接嵌进 DP 微调管线，不像 FO 那样要额外注入高斯噪声。

**3. 去混淆评测：剥掉 task alignment 才看得见 ZO 的真本事**

position paper 要立得住，得给出"现状失灵"的可观察证据，这条就是那把校准尺。作者要求所有 ZO 评测同时报告两种设置——带 task alignment（用 prompt 把下游任务对齐到预训练目标）和不带——并把前向梯度方法 $f'(\mathbf{x};\mathbf{u})\mathbf{u}$ 列为强制基线。前向梯度的地位很特殊：当 $\mu \to 0$ 时有限差分收敛到方向导数 $f'(\mathbf{x};\mathbf{u}) = \mathbf{u}^\top \nabla_{\mathbf{x}} f(\mathbf{x})$，它用一次 JVP 就能精确拿到，结构上是 ZO 估计器的"无噪上界"。有了它就能分清责任：前向梯度都做不动，卡的是任务难度而非 ZO；前向梯度能做而 ZO 做不动，才是估计器真的不行。

task alignment 的隐患在于它把下游任务简化到贴近预训练分布，让 ZO 在这种"易学化"情形下显得格外强。论文用 Gemma2-2B 在 SST-2 / RTE / WiC 上跑 MeZO / Sparse-MeZO / HiZOO / LOZO 四个 stateful ZO 方法做对比（图 2）：去掉 alignment 后大多数方法都明显掉点，方法间的相对排名甚至反转——这说明现行协议混淆了"ZO 的优化能力"和"任务被简化的程度"。换句话说，任何号称更优的 ZO 方法，先得证明自己离这个无噪上界有多远，以及在非对齐场景里还剩多少优势。

## 实验关键数据

position paper 只有一组验证性实验（图 2），用来给 P6 站台。

### 主实验：task alignment 对 ZO 性能的混淆效应

| 任务 | 方法 | w/ alignment | w/o alignment | 趋势 |
|------|------|--------------|---------------|------|
| SST-2 / RTE / WiC | MeZO | 较高 | 显著下降 | 普遍掉点 |
| SST-2 / RTE / WiC | Sparse-MeZO | 较高 | 显著下降 | 普遍掉点 |
| SST-2 / RTE / WiC | HiZOO | 较高 | 显著下降 | 普遍掉点 |
| SST-2 / RTE / WiC | LOZO | 较高 | 显著下降 | 普遍掉点 |

> 论文以 bar chart 形式呈现，未列具体数值；定性结论是 4 个 stateful ZO 方法在 Gemma2-2B 上跨三个 GLUE 任务都出现显著掉点，且相对排名在两个 setting 之间发生反转。

### 文献调研表：现有 ZO 工作对 P1/P2/P3 的覆盖

| 代表方法 | 使用场景 | P1（方差控制） | P2（查询代价） | P3（前向梯度基线） |
|----------|---------|----------------|----------------|---------------------|
| ZO-NP / AdaZO / PaZO / Sparse-MeZO / PseuZO / PAZO / OPZO / HiSo | U1 微调 或 U2 从零训练 | ✓ | ✗ | ✗ |
| SharpZO | U1 | ✓ | 部分 | ✗ |
| FZOO | U1 | ✗ | ✓ | ✗ |

### 关键发现

- **task alignment 是当前 ZO "好成绩"的隐藏功臣**：在去掉 prompt 对齐后，所有四个 stateful ZO 方法都明显掉点，且相对排名重排，说明现行 leaderboard 在很大程度上反映的是任务被简化的程度而非 ZO 算法本身的优化能力。
- **社区把 99% 的精力都花在 P1**：表 1 中 10 篇代表性工作几乎只解决方差控制问题（P1），查询代价（P2）和前向梯度基线（P3）这两条几乎被全员忽视；唯一同时考虑 P3 的是 ZO-Bench（Zhang et al. 2024c）。
- **U2（从零训练）是最缺人的方向**：现有 ZO 工作绝大多数是 U1（pretrained model 微调），U2 场景下查询代价是真正的瓶颈，需要把 P2 摆在最前面。

## 亮点与洞察

- **把 ZO 重新定义为"推理型负载"是最大的思想跳跃**：作者指出 ZO 的工作画像和 RL rollout / serving 阶段一模一样，因此应当跑在 vLLM / FlashAttention / PagedAttention 这类 inference 栈上而不是 DeepSpeed / Megatron / FSDP——这一刀切下去把整个 ZO 系统栈的优化方向从"压缩反向"翻转到"加速前向"，是教科书级的范式重命名。
- **S-RGE 的几何诠释把方差控制和子空间学习焊在一起**：$\mathbb{E}[\hat{\nabla} f] = \mathbf{PP}^\top \nabla f$ 这一行公式把"低秩梯度先验"、"投影偏差"、"方差–维度脱钩"三件事统一起来，给后续工作（混合 FO–ZO、谱域 ZO）提供了非常干净的接口。
- **共享种子 + 标量传输的通信协议**几乎不需要新算法就能让 ZO 在联邦/分布式场景秒变 communication-optimal，是个"实现成本极低、影响极大"的可迁移 trick。
- **"前向梯度作为强制基线"**是这篇 position paper 给出的最具操作性的评测建议——以后任何 ZO 论文都应当把前向梯度方法（PyTorch 一行 JVP）作为对照组报出来，否则没法判断方法到底强在哪里。
- **DP 论证非常优雅**：把 ZO 的"内在噪声"直接对接到 differential privacy 的"加噪要求"，让 ZO 在 DP fine-tuning 场景下几乎是"白嫖"隐私预算。

## 局限与展望

- **作者承认的局限**：S-RGE 的有效性建立在"模型梯度本身低秩"这一经验观察上，并非所有架构和任务都满足；ZO-native 系统栈尚不存在，所有论述都停留在原理层面，没有跑出实际的端到端吞吐量数字。
- **自己发现的局限**：作为 position paper，唯一的实验（图 2）规模较小（Gemma2-2B + GLUE 子任务），且只给出柱状图而无具体数值表，论据的强度更多依赖文献综述而非实证；表 1 对 P1/P2/P3 的打钩判定可能因评审标准而争议（"partially" 这种打分缺乏量化定义）；P4 推崇的"子空间 + 谱域"组合目前只在小模型上有早期证据（Muon、LOZO 等），还没有证明在 7B+ 规模下还能保住方差红利。
- **改进思路**：一是补一个"在固定 query budget 下，FO vs forward-gradient vs ZO"的统一对比 benchmark，把 P3 直接做成可复用的工具；二是把 ZO-native pipeline schedule 在真实 inference 引擎（如 vLLM）里落地一个 prototype，验证近零气泡是否真的能让大模型 ZO 训练接近推理吞吐；三是把生成式估计器（A3 提到的 DM-based gradient denoiser）从概念落到代码，用 ControlNet-style conditional DM 学一个"从 RGE 噪声梯度去噪到 FO 梯度"的网络，看是否真能突破 $O(d)$ 方差墙。

## 相关工作与启发

- **vs MeZO (Malladi et al. 2023)**：MeZO 是 ZO 在 LLM 微调上的爆点工作，但它正是 P6 所批评的"严重依赖 task alignment"的代表；本文把它收编为 ZO 复兴的契机而非终点，认为社区后续过度聚焦于沿 MeZO 路线做估计器改良（Sparse-MeZO / HiZOO / PaZO 等），却忽略了子空间、系统级和评测级的红利。
- **vs ZO-Bench (Zhang et al. 2024c)**：ZO-Bench 是少数把前向梯度纳入基线的工作，本文把它的方法论提升到"通用评测协议"的高度（P3+P6+A1），呼吁把 forward gradient 列为强制对照组。
- **vs Forward Gradient (Baydin et al. 2022; Ren et al. 2023)**：前向梯度系列原本是 ZO 的"近亲对手"，本文不是去贬低它，而是把它升格为校准尺——它是 ZO 估计器精度的天然上界，能帮助判断瓶颈到底在 ZO 算法还是任务本身。
- **vs Distributed FO (DeepSpeed / Megatron / FSDP)**：作者明确反对把 ZO 算法移植进 FO 系统栈，因为这些系统的设计权衡（重计算换显存、tensor 并行换吞吐）会反过来放大 ZO 的计算成本；这种"系统栈不匹配"的论述是 ZO 社区第一次被点名。
- **vs Local Learning / Bio-plausible BP-free (Hinton 2022; Nøkland 2016 等)**：那条线追求生物可解释性，但 scalability 比 ZO 还差；本文把 ZO 摆在"既保留可解释性又能上规模"的中间位置。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是新算法，但 6 条 position 中 P4（子空间+谱）和 P5（推理型负载 + 单向 pipeline）属于把 ZO 从优化问题重新定义为系统/谱学问题的非平凡视角切换。
- 实验充分度: ⭐⭐⭐ 仅一组小规模实证（Gemma2-2B 上 4 方法 × 3 任务的 alignment 对比），符合 position paper 的体量但缺乏定量数值表。
- 写作质量: ⭐⭐⭐⭐⭐ 结构极清晰（P1–P3 立可行性边界，P4–P6 立新方向，A1–A5 落地），每条 position 都用同一个 RGE 数学骨架推出来，立论 + 反驳 + 评测证据齐全。
- 价值: ⭐⭐⭐⭐⭐ 对 ZO 社区是一次必要的视角清算——尤其"前向梯度作为强制基线"、"ZO 跑在推理栈"、"去 task alignment 评测"这三条都是可立即采纳的方法论改良。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Position: Ideas Should be the Center of Machine Learning Research](position_ideas_should_be_the_center_of_machine_learning_research.md)
- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](../../ICLR2026/interpretability/behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[ICML 2026\] A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments](a_deep_learning_model_of_mental_rotation_informed_by_interactive_vr_experiments.md)
- [\[ICML 2026\] Expand Neurons, Not Parameters](expand_neurons_not_parameters.md)
- [\[ICML 2026\] Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions](physics_from_video_identifiability_of_time-invariant_second-order_odes_under_min.md)

</div>

<!-- RELATED:END -->
