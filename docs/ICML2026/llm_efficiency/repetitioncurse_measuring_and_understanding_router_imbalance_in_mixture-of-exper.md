---
title: >-
  [论文解读] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress
description: >-
  [ICML 2026][LLM效率][MoE路由失衡] 通过给 MoE 大模型喂"同一个 token 重复 N 遍"这种极简的 OOD 提示，作者发现 router 会把几乎所有 token 路由到固定的少数几个 top-$k$ 专家上，从而在专家并行（EP）部署下制造单卡瓶颈、把别的 GPU 全部 idle 住，在 8-GPU 集群上把 TTFT 拉高 20%–148%，把 MoE 的并行加速器反过来变成 DoS 攻击面。
tags:
  - "ICML 2026"
  - "LLM效率"
  - "MoE路由失衡"
  - "专家并行"
  - "DoS攻击"
  - "TTFT"
  - "黑盒攻击"
---

# RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress

**会议**: ICML 2026  
**arXiv**: [2512.23995](https://arxiv.org/abs/2512.23995)  
**代码**: 待确认  
**领域**: LLM效率 / MoE系统 / 推理安全  
**关键词**: MoE路由失衡, 专家并行, DoS攻击, TTFT, 黑盒攻击

## 一句话总结
通过给 MoE 大模型喂"同一个 token 重复 N 遍"这种极简的 OOD 提示，作者发现 router 会把几乎所有 token 路由到固定的少数几个 top-$k$ 专家上，从而在专家并行（EP）部署下制造单卡瓶颈、把别的 GPU 全部 idle 住，在 8-GPU 集群上把 TTFT 拉高 20%–148%，把 MoE 的并行加速器反过来变成 DoS 攻击面。

## 研究背景与动机

**领域现状**：现代 LLM 普遍用 MoE（Mixtral / Qwen3-MoE / DeepSeek-V3 / GPT-OSS / Llama-4-Scout 等）扩大容量却不放大推理成本，工业部署上配套的是 **专家并行（Expert Parallelism, EP）**——把不同专家放到不同 GPU 上，靠 router 决定每个 token 去哪几个专家，从而省显存、省通信。vLLM、SGLang 等主流引擎都默认这套路线。

**现有痛点**：MoE 在**训练阶段**会加 expert-/device-level balance loss 强制均衡，但**推理阶段没有任何均衡约束**。如果某个 batch 里 token 不均匀地集中到少数专家，对应 GPU 会变 straggler，其他 GPU 必须 idle 等它做完 all-reduce，等于把"并行加速"退化成"串行最慢一项"。

**核心矛盾**：MoE 系统的 **效率假设**（token 在专家间均匀分布）和 **对抗鲁棒性** 之间存在天然矛盾——一旦攻击者能制造"路由坍缩"，EP 越大反而被害越深。已有 LLM-DoS 工作要么逼模型生成超长输出（攻击者得为每个 token 付费），要么靠 backdoor / prompt injection，没人正面攻击 MoE 的 router 本身。

**本文目标**：(1) 在严格黑盒（不知模型权重、不知路由策略、不知 expert-GPU 映射）下找到能可靠造成路由失衡的攻击 prompt；(2) 量化这类攻击对 TTFT 和 SLA P$_{99}$ 的实际危害；(3) 系统刻画哪些架构 / 部署因子放大或缓解该漏洞。

**切入角度**：作者从 embedding 空间观察：router 是个对 hidden state 的确定性函数 $G(h)=\text{Softmax}(h\cdot W_{\text{router}})$，要逼 router 选同一组 top-$k$ 专家，本质上等价于让相邻 token 的 hidden state **塌缩到同一簇**——也就是让每层 embedding 方差 $D(H^l(X))=\frac{1}{N}\sum_i\|h^l_i-\bar h^l\|_2^2$ 尽量小。

**核心 idea**：与其用白盒梯度优化 $\arg\min_X \sum_l D(H^l(X))$，不如做**最暴力的零阶近似**——直接让输入是同一个 token 重复 N 遍（RepetitionCurse），同样能把 hidden state 压塌、把路由打爆，且不需要任何模型内部信息，也能通过简单改首 token 绕开 KV cache 复用。

## 方法详解

### 整体框架
威胁模型是典型的黑盒 API 攻击：服务方在多 GPU 集群上用 EP + prefill-decoding disaggregation 部署 MoE，攻击者只能从公开 API 发 prompt，不关心模型输出质量，只追求把 **legitimate user 同 batch 里的 TTFT** 撑爆。RepetitionCurse 的 pipeline 极短：(1) 从词表挑一个 token $t$；(2) 构造 prompt $P_t = [t, t, \dots, t]$（除去 chat template / system prompt 部分）；(3) 直接发送即可。攻击效果由 router 内部坍缩 + EP 调度共同放大。

### 关键设计

**1. 基于 embedding-variance 最小化的攻击目标：把"逼 router 坍缩"写成一个可分析的优化问题**

要解释"重复 token 为什么管用"，得先把攻击目标形式化。作者定义层级 embedding 方差 $D(H^l(X))=\frac1N\sum_i\|h^l_i-\bar h^l\|_2^2$，把最优攻击 prompt 写成 $X^*=\arg\min_X\sum_{l=1}^L D(H^l(X))$——方差越小，相邻 token 的 hidden state 越塌缩到同一簇，作为 hidden state 确定性函数的 router $G(h)=\text{Softmax}(h\cdot W_{\text{router}})$ 就会把它们全推向同一组 top-$k$ 专家。直接解这个 $\arg\min$ 需要白盒梯度，但作者证明：让所有 token 完全相同 $\Rightarrow$ 相邻 hidden state 差异最小 $\Rightarrow$ 经验上同时压低 embedding variance 与 embedding entropy（参 Skean et al., 2025），于是白盒优化退化成"选哪个 token + 重复多少次"两个超参，黑盒可行。这条理论同时点破了漏洞的根因：MoE router 的均衡能力**隐式假设输入 token 在 embedding 空间足够发散**，一旦打破这个假设，后训练怎么微调都补不回来——这正解释了为什么 base 和 instruct 变体上漏洞如出一辙。

**2. 理论最大失衡 TMI：给出单卡过载的上界，并量化"EP 越大越脆弱"**

有了攻击目标还要知道它最坏能造成多大破坏。设每卡放 $E_d=|\mathcal{M}_l(d)|$ 个专家、router 取 top-$k$，则被打爆那张卡的最坏负载是 $\min(k,E_d)$，而公平份额只有 $k/|\mathcal{D}|$，两者之比就是理论最大失衡 $\text{TMI}=\dfrac{\min(k,E_d)}{k/|\mathcal{D}|}$。这个式子呈两段行为：当 $k\le E_d$（稀疏型如 DeepSeek-V3）时攻击随 EP size 线性放大，理论上能制造"完美瓶颈"；当 $k>E_d$（如 Mixtral-8x7B 在 EP=8 时每卡只剩 $E_d=1$ 个专家、却要选 $k=2$）就被卡在 $|\mathcal{D}|\cdot E_d/k=4$ 倍而非满血的 8 倍。TMI 的意义在于把"运维侧放大 EP 提效率"和"安全侧放大攻击面"绑成同一个 trade-off，给出一条可执行的部署建议：不要无脑追求大 EP。

**3. 黑盒实战的两条务实降级：承认达不到 TMI 上界，但仍普适有效**

RepetitionCurse 在真实黑盒里跑不满 TMI 理论值，作者诚实地定位了两个 gap 并各自处理。其一是**对 Expert-GPU 映射不可知**：动态均衡器（如 DeepSeek 的 EPLB）每隔 ~10 min 才重排一次且开销不小，重排窗口内映射是静止的，作者于是沿用 vLLM/SGLang 默认的"按序分配"映射作为分析基线。其二是**无法指定目标专家**：攻击能把 token 集中到某组 top-$k$ 专家，却不能挑是哪 $k$ 个，万一命中的两个专家恰好被分到不同 GPU（如 Mixtral 在 EP=2 的某些层），延迟收益就消失了；作者把这一限制建模成一个概率因子，在跨词表、跨层的统计意义上仍能稳定拿到 1.07×–2.48× 的 TTFT 放大。这两条降级反而凸显了真正的威胁来源——不是某个精心构造的最优白盒攻击，而是这种**普适但不完美**的黑盒能力：攻击者只要把词表扫一遍，总能找到对当前部署"恰好命中"的那个 token。

### 训练策略
本文是**零阶黑盒攻击 + 系统测量**，没有任何可训练参数，也不需要梯度或 fine-tune，攻击只有两个超参：用哪个 token、prompt 多长（用 batch 内长度比例 $\alpha\in\{\tfrac12,1\}$ 表示，对应表 2 的两列）。真正的工作量都在测量侧：自动扫 139 个 HuggingFace MoE 配置，并在 13 个代表性模型上跑 EP=2/4/8/16/32 的全网格 benchmark。

## 实验关键数据

### 主实验
覆盖 13 个 MoE：4 个 Mixtral 系（$E{=}8,k{=}2$）+ Qwen3-30B-A3B 三件套（$E{=}128,k{=}8$）+ GPT-OSS-20B/120B + Llama-4-Scout-17B + DeepSeek-V2-Lite + Kimi-Linear 两件套。指标用作者新定义的 **LAR**（Latency Amplification Ratio）：$\text{LAR}_{\text{moe}}$ 是单层 MoE 计算延迟放大倍数，$\text{LAR}_{\text{ttft}}$ 是端到端 TTFT 放大倍数。

| 模型 | EP size | $\text{LAR}_{\text{moe}}$ ($\alpha{=}\tfrac12/1$) | $\text{LAR}_{\text{ttft}}$ ($\alpha{=}\tfrac12/1$) | 备注 |
|------|---------|---------------------------|---------------------------|------|
| Mixtral-8x7B | 8 | **2.01 / 2.68** | **1.61 / 2.48** | 经典 8-GPU 部署，TTFT 拉到 2.48× |
| Mixtral-8x7B-It | 8 | 1.94 / 3.12 | 1.65 / 2.48 | instruct 变体同样脆弱 |
| Qwen3-30B-A3B | 32 | 2.28 / 3.22 | 1.53 / 2.15 | 高稀疏模型在大 EP 下持续放大 |
| Qwen3-Coder-30B-A3B-It | 32 | 2.32 / 3.04 | 1.51 / 2.08 | 代码专用版也受害 |
| GPT-OSS-20B | 8 | 1.20 / 1.46 | （文中片段未给完整端到端） | 小 EE/小 $k$ 模型较抗压 |

定性结论：(a) Mixtral 系在常用 8-GPU EP 下 TTFT 普遍放大 **1.29×–2.48×**；(b) 跨 13 个模型、Qwen3/GPT-OSS/DeepSeek 等不同架构、含线性 attention 的 Kimi-Linear 全部命中漏洞；(c) SLA 影响：P$_{99}$ TTFT < 20s 的违约率从基线 1% 上升到 **1.4%–13.6%**，足以触发 SLA 罚款和不必要的 autoscaling。

### 消融 / 分析实验

| 维度 | 关键结果 | 说明 |
|------|---------|------|
| 词表覆盖率 $\mathcal{B}$ | EP=$E$ 时 $\mathcal{B}\to 1$ | "随便挑一个 token 重复"几乎总能造成路由集中 |
| EP size 扫描 | $\mathcal{B}$ 随 EP 单调上升 | EP 越大越脆弱，与 TMI 公式定性一致 |
| 模型族内一致性 | Mixtral-8x7B 与其指令/中文/Nous 微调版 coverage 几乎相同 | 漏洞在**预训练阶段**已经埋下，后训练改不掉 |
| 宽 vs 深 MoE | 大 $E$ / 小 $L$ 的 Qwen3-MoE 比深窄 Mixtral 抗性更好 | 同算力下"宽而浅"是更安全的 MoE 设计 |
| Expert-GPU 映射偏置（图 3） | 自然文本下 router 近均匀，RepetitionCurse 下极端集中 | 验证攻击根因不在硬件而在 router 行为 |

### 关键发现
- 最强单点：Mixtral-8x7B-It 在 EP=8、$\alpha{=}1$ 时单层 MoE 延迟放大 **3.12×**、TTFT 放大 **2.48×**，对应 8-GPU 集群上 prefill 阶段几乎被卡成单卡。
- 最反直觉的设计权衡：**EP size 越大效率越高，但攻击面也越大**。作者据此建议在没有更好推理时均衡策略前，**主动限制 EP size**（与业界一味追求大 EP 的趋势相反）。
- 攻击 prompt 极廉价：无梯度、无白盒、无超长生成（攻击者不用为输出 token 付费），且能通过修改首 token 绕过 KV cache 复用，性价比远高于已有 LLM-DoS 攻击。
- 防御启示：动态映射器（EPLB）的 10 分钟级窗口对秒级 SLA 几乎无效；要么把均衡 loss 搬进推理时，要么 batch-level 检测重复模式。

## 亮点与洞察
- 把"训练-推理目标不一致"这一系统性 gap 提炼成可量化的攻击向量：训练时强制 load balance、推理时却完全松绑，攻击者只要让 OOD 输入触发 router 失衡就能反过来用 EP 害死系统。
- 攻击形式极致简化——"重复同一个 token"——但作者用 embedding variance + TMI 两条理论把它和最优白盒攻击挂钩，让"看起来像 hack"的方法获得严肃的理论解释。
- 新引入 LAR（MoE 层延迟放大率、TTFT 放大率）和 bottleneck coverage $\mathcal{B}$ 两个指标，把"路由失衡"从模型侧概念翻译成"运维 SLA 侧"语言，对推理系统社区直接可用。
- 系统性扫 139 个 HF MoE 配置 + 13 个跑得起来的 SOTA，给出"宽而浅" vs "深而窄"、base vs instruct、稀疏度 vs EP size 等多个维度的实证规律，可作为后续 MoE 推理引擎设计的 checklist。

## 局限与展望
- 攻击不能挑专家：只能保证集中到 top-$k$，不能控制具体是哪 $k$ 个；当 EPLB 等动态均衡足够频繁时危害会被显著缓解（但作者也指出 10 min 窗口太长）。
- 实验主要在 prefill-bound 场景刻画 TTFT，对 decoding-bound 长输出场景的影响、以及 chunked-prefill / speculative decoding 等新调度策略下的鲁棒性尚未系统评估。
- 防御侧本文只给出"限 EP size"这一保守建议，没有提出 router-side 的对抗均衡器、token-level 重复检测器、或基于 dispatch 监控的在线缓解；这是顺理成章的后续方向。
- 13 个模型已覆盖主流但都是 $\le 120$B 的开源 MoE，DeepSeek-V3 这种超大 EP（32 prefill）真实集群上能否复现 TMI 量级的放大仍待 vendor 侧验证。

## 相关工作与启发
- **vs Gao et al. 2024 / Zhang et al. 2024（长输出型 LLM DoS）**：他们逼模型生成到 max tokens 来榨 backend 资源，攻击者需要为每个生成 token 付费；本文只用极短输入就压跨 prefill 阶段，性价比和隐蔽性都更高。
- **vs Li et al. 2025b（"无尽思考"型推理攻击）**：那条线针对带 reasoning 的模型让其陷入死循环，攻击的是 token 数；本文攻击的是 **每个 token 的硬件利用率本身**，与 reasoning 能力正交，对所有 MoE 推理服务都有效。
- **vs EPLB / DeepSeek-V3 部署经验（DeepSeek-AI 2024b/2025）**：EPLB 是 vendor 侧针对负载漂移的动态映射；本文揭示其重排周期对秒级 SLA 攻击远远不够，等价于给 EPLB 列了一个新的对抗 benchmark。
- **vs grouped GEMM / vLLM / SGLang 等 MoE kernel 工作**：以往优化目标是吞吐与延迟，本文提示"对抗鲁棒性"必须成为第三个一等公民，否则越优化反而越脆。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把 MoE router 失衡正式建模为 DoS 攻击向量，攻击方法又特别简洁。
- 实验充分度: ⭐⭐⭐⭐ 13 个模型 × 5 个 EP size 配合 139 个 HF 配置统计调研，已足够说服力，缺真实超大集群（DeepSeek-V3 级）实测。
- 写作质量: ⭐⭐⭐⭐ 把系统视角、对抗视角、理论 TMI 串得很顺，符号略多但结构清晰。
- 价值: ⭐⭐⭐⭐⭐ 对所有部署 MoE 的厂商立刻可执行（限 EP size、加 router 监控），并为 MoE 推理系统打开"对抗鲁棒性"这一新坑。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs](softmoe_soft_differentiable_routing_for_mixture-of-experts_in_llms.md)
- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[ICML 2026\] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)

</div>

<!-- RELATED:END -->
