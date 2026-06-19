---
title: >-
  [论文解读] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving
description: >-
  [ICML 2026][LLM效率][AFD] 本文为新兴的 Attention-FFN 解耦 (AFD) 推理架构提供首个理论框架,基于"prefill 长度有限均值 + decode 长度服从几何分布"的概率工作负载模型,推导出 rA-1F 拓扑下最优 A/F 比的闭式解 $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$,并用 trace-calibrated 模拟器验证理论与实测最优值偏差 <10%。
tags:
  - "ICML 2026"
  - "LLM效率"
  - "AFD"
  - "A/F ratio"
  - "双层优化"
  - "几何分布"
  - "roofline 模型"
---

# Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving

**会议**: ICML 2026  
**arXiv**: [2601.21351](https://arxiv.org/abs/2601.21351)  
**代码**: 有 ([anonymous.4open.science/r/AF-release-1C11](https://anonymous.4open.science/r/AF-release-1C11))  
**领域**: LLM效率 / 推理系统 / Attention-FFN 解耦  
**关键词**: AFD, A/F ratio, 双层优化, 几何分布, roofline 模型

## 一句话总结
本文为新兴的 Attention-FFN 解耦 (AFD) 推理架构提供首个理论框架,基于"prefill 长度有限均值 + decode 长度服从几何分布"的概率工作负载模型,推导出 rA-1F 拓扑下最优 A/F 比的闭式解 $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$,并用 trace-calibrated 模拟器验证理论与实测最优值偏差 <10%。

## 研究背景与动机

**领域现状**:LLM 推理服务从 monolithic 架构演进到 disaggregation:先有 **PD 解耦** (Prefill compute-bound 与 Decode memory-bound 分离,Zhong et al. 2024),近期又出现 **AFD (Attention-FFN Disaggregation)** —— 注意到 decode phase 内部 Attention (stateful + memory-bound,被 KV cache 读取主导) 和 FFN (stateless + 加 batch 后 compute-bound) 计算特征也不同,把它们分开部署到不同硬件池,让多个 Attention 实例共享一个 FFN 实例 (rA-1F 拓扑)。

**现有痛点**:AFD 的性能对 A/F 比 $r$ 极其敏感 —— $r$ 太小 FFN 闲死等数据,$r$ 太大 Attention 实例堵在等 FFN。现有 AFD 系统 (Wang et al. 2025, Zhu et al. 2025, Zuo et al. 2025) **靠经验搜索**配 $r$,既没理论指导也不知道"最优在哪、为什么"。

**核心矛盾**:**Attention 工作量是 non-stationary 的** —— KV cache 每步增长、完成的请求被新请求替换 (continuous batching),所以 $T_k$ 是随时间漂移的随机过程;而 **FFN 工作量是 stable 的** (只看 batch size)。这导致静态 microbatch schedule 永远无法持续最优,必然产生 pipeline bubble。要选 $r^*$ 必须先把这个非平稳随机动态化简成可优化的标量。

**本文目标**:(1) 建立能捕捉 microbatch pipelining + 同步 barrier + continuous batching 的概率工作负载模型;(2) 推导最优 A/F 比的闭式解;(3) 用模拟器验证理论的预测能力。

**切入角度**:作者注意到生产 LLM trace 里 decode length 高度符合**几何分布** $D\sim \text{Geo}(p)$ —— 这个 memoryless 性质让 $X_b(k)$ (slot 是否继续) 独立于 $i_b(k)$ (当前 decode 索引),从而把复杂的 non-stationary 过程化成可解的 Markov 链;预训练长度 $P$ 只需均值 $\mu_P$,不需要具体分布。

**核心 idea**:用"horizon-average token load $\bar{T}=B(\mu_P+\mu_D)$"代替瞬时 $T_k$,把 $\tau=\max\{t_A, t_C, t_F\}$ 的 cycle time 模型拆成三个 regime (Attention / Communication / FFN 瓶颈),分别求最优后取最大就是全局 $r^*$。

## 方法详解

### 整体框架
模型 AFD bundle 为 rA-1F 拓扑,每个 Attention 实例维护 $B$ 个 slot,decode 步分四阶段:Attention 计算 → A→F 通信 → FFN 处理 $rB$ 聚合 batch → F→A 通信。Cycle time $\tau(B;r)=\max\{t_A(T), t_C(B), t_F(rB)\}$,目标是最大化 per-instance throughput $\text{Throughput}=\frac{1}{r+1}\cdot \frac{rB}{\tau(B;r)}$。先用概率分析把 $T$ 替换为 horizon-average 值 $\bar{T}$,再做 regime 分析。

### 关键设计

1. **概率工作负载模型 + 几何分布关键洞察**:

    - 功能:把 Attention-side 的非平稳随机动态化简成可解析的期望递推
    - 核心思路:用 $X_b(k)\sim \text{Bernoulli}(1-p)$ 表示 slot $b$ 在 step $k$ 是否继续,decode index 更新 $i_b(k+1)=X_b(k)\cdot(i_b(k)+1)$,prefill 长度 $s_b(k+1)=X_b(k)\cdot s_b(k)+(1-X_b(k))\cdot S_b'(k)$;**几何分布的 memoryless 性质**保证 $X_b(k)$ 独立于 $i_b(k)$,从而推出干净的期望递推:$\mathbb{E}[P_k]=B\mu_P$ (常数),$\mathbb{E}[D_k]=B\frac{1-p}{p}(1-(1-p)^k)$ (从 0 指数 saturate 到 $B\mu_D$),所以 $\mathbb{E}[T_k]=B\mu_P+B\frac{1-p}{p}(1-(1-p)^k)$
    - 设计动机:几何分布不是数学方便而已 —— 它**反映了 LLM autoregressive 生成的真实物理**:每步以近似常数概率产 EOS,与已生成 token 数无关;作者用 SGLang、AzureLLM 等生产 trace 验证了 decode length 高度服从几何 (Figure 3);prefill 长度只取均值 $\mu_P$ 让模型不依赖具体分布,鲁棒性高

2. **Horizon-Average Token Load 的大数律收敛**:

    - 功能:把"随时间漂移的 $\mathbb{E}[T_k]$"压成一个标量代表性工作量,可以塞进优化目标
    - 核心思路:定义 horizon-average $\bar{T}(B;N):=\frac{1}{K(B)}\sum_{k=0}^{K(B)-1}\mathbb{E}[T_k]$,其中 $K(B)=N/(Bp)$ 是服务 $N$ 个请求的期望步数;Proposition 4.3 证明 $N\to\infty$ 时 $\bar{T}\to (\mu_P+\frac{1-p}{p})B = B(\mu_P+\mu_D)$ —— 也就是"平均每个 slot 的总长 = 期望 prefill + 期望 decode"这一直觉的严格版本
    - 设计动机:这一步是优化可解的关键 —— 直接用 $\mathbb{E}[T_k]$ 进 max-min 优化无法得闭式解;退到 horizon average 既保留了 long-run 行为又能 close form;大数律给了渐近正确性保证而不是 hand-wave 近似

3. **三 Regime 分析与全局最优 $r^*$ 闭式解**:

    - 功能:把 $\tau=\max\{t_A, t_C, t_F\}$ 这个 piecewise function 拆开,每段单独求最优,组合得全局最优
    - 核心思路:定义 $\bar{t}_A=\alpha_A\bar{T}+\beta_A$、$\bar{t}_C=\alpha_C B+\beta_C$、$\bar{t}_F(r)=\alpha_F rB+\beta_F$。**Regime I (Attention-bottleneck)** $r\leq r_A:=(\bar{t}_A-\beta_F)/(\alpha_F B)$ 时 throughput $\propto r/(r+1)$ 递增,最优在 $r_A$;**Regime II (Comm-bottleneck)** $r\leq r_C:=(\bar{t}_C-\beta_F)/(\alpha_F B)$ 同理最优在 $r_C$;**Regime III (FFN-bottleneck)** $r\geq r_{\text{crit}}$ 时 throughput $f(r)=rB/[(r+1)(\alpha_F rB+\beta_F)]$ 是 unimodal 的,求导 $f'(r)=0$ 解出 $r_{\text{peak}}=\sqrt{\beta_F/(\alpha_F B)}$。最终 **Theorem 4.4** 给出 $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$,即三个 regime 各自最优的取大
    - 设计动机:max 形式来自三个 regime 都希望 "$r$ 尽量大但不要让自己变成瓶颈" —— 取最大就是恰好不让任何一个组件成为 binding bottleneck;$r_{\text{peak}}=\sqrt{\beta_F/(\alpha_F B)}$ 揭示了"FFN bottleneck 下 $r$ 与 $\sqrt{1/B}$ 成比例" —— batch 越大,需要的 attention 实例越少,符合直觉

### 损失函数 / 训练策略
**纯系统理论,无训练**。计算流程:(1) 给定硬件参数 $(\alpha_A, \beta_A, \alpha_F, \beta_F, \alpha_C, \beta_C)$ 和工作负载 $(\mu_P, \mu_D)$;(2) 算 $\bar{T}\approx B(\mu_P+\mu_D)$;(3) 算 $r_A, r_C, r_{\text{peak}}$;(4) $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$。参数标定:DeepSeek-V3 + 华为 Ascend 910C NPU trace 线性回归。

## 实验关键数据

### 主实验:理论 $r^*$ vs 模拟最优 (DeepSeek-V3 + Ascend 910C)

| 工作负载配置 | 理论 $r^*$ | 模拟最优 | 相对误差 |
|---------------|-----------|----------|----------|
| 典型 chat ($\mu_P$=200, $\mu_D$=300, $B$=32) | (理论值) | (模拟值) | <10% |
| 长上下文 ($\mu_P$=2000, $B$=16) | (上升) | (匹配) | <10% |
| 短回复 ($\mu_D$=50) | (下降) | (匹配) | <10% |

跨多种 (batch size $B$, 上下文长度 $\mu_P$) 组合,理论 $r^*$ 与模拟器穷举出来的最优值偏差始终 <10%。

### 关键发现 (摘自论文 ablation)

| 配置 | 趋势 | 解释 |
|------|------|------|
| Batch size $B$ ↑ | 最优 $r^*$ ↑ | $r_A$ 项随 $\bar{T}=B(\mu_P+\mu_D)$ 增长 |
| Context length $\mu_P+\mu_D$ ↑ | 最优 $r^*$ ↑ | Attention 工作量上升,需更多 attention 实例 |
| 走 FFN-bottleneck regime 时 | $r^*=r_{\text{peak}}=\sqrt{\beta_F/(\alpha_F B)}$ | $r^* \propto 1/\sqrt{B}$,$B$ 大反而 $r^*$ 小 |
| 走 Attention-bottleneck regime 时 | $r^*=r_A$ 线性增长 | KV cache 主导,$r$ 必须跟上 token load |

### 关键发现
- **几何分布是 modelling 的关键工程选择**:作者用真实 trace 验证 (Figure 3) 而不是假设,且 memoryless 性质恰好让递推可解 —— 这是"经验观察 + 数学便利性恰好对上"的漂亮例子
- **三 regime 视角让"为什么 $r$ 该设这个值"变得可解释**:运维不用再瞎试,可以根据 $\bar{t}_A, \bar{t}_C, \bar{t}_F$ 的大小关系判断系统在哪个 regime,据此推出 $r^*$
- **$r^*$ 与 $\sqrt{1/B}$ 的关系反直觉但符合 trade-off**:在 FFN-bottleneck regime 下,batch 越大,每个 FFN 周期能处理越多 token,需要的 attention 实例反而越少 —— 揭示了大 batch 时不该盲目堆 attention
- **<10% 误差跨多个工作负载稳定**:这说明 horizon-average 近似确实抓住了真实动态的关键

## 亮点与洞察
- **"几何分布 + 大数律 + 三 regime"三件套**:把一个看起来要靠 simulation 调参的非平稳排队问题,化简成一个三分支取 max 的闭式公式 —— 这是把统计物理 + 排队论 + 系统工程拼合的漂亮范例,后续做 PD-AFD-MoE 等更复杂解耦拓扑都可以借鉴
- **Roofline 模型 + 线性 latency 是 LLM 推理建模的标准抽象**:作者明确把 $t_A=\alpha_A T+\beta_A$ 等线性模型挂在 roofline + 已有 LLM serving 文献 (Yuan et al. 2024) 上,既严谨又可移植到其他硬件 (附录 B 给了通用 derivation framework)
- **诚实承认"模拟验证而非真实部署"**:作者坦白说 AFD 还没有 mature 的开源实现,所以只在 trace-calibrated 模拟器上验证 —— 这种"理论先行、为系统设计提供蓝图"的姿态比"硬上 prototype"更负责任

## 局限与展望
- 只验证了模拟器,没在真实 AFD 系统上跑 —— 模拟器虽然 trace-calibrated 但仍可能漏掉某些硬件细节 (NUMA、网络抖动、调度器开销)
- 假设 prefill 长度只需均值,但实际重尾分布 (热门长 prompt) 可能让方差不可忽视
- 几何分布在不同任务/模型上 $p$ 不同,如果同一服务混合多种工作负载 (chat + code + 长写作),$\mu_D$ 取均值可能不够
- 没考虑 SLA / TPOT 约束 —— 实际部署 throughput 不能无脑最大化,还要满足 P99 latency
- 没考虑能耗 / 成本约束,只看 throughput per instance
- 未来方向:扩展到 PD-AFD 联合解耦、MoE 模型的 Expert-FFN 分组、heterogeneous 硬件 (mixed GPU/NPU) 配比

## 相关工作与启发
- **vs PD disaggregation (Zhong et al. 2024, Patel et al. 2024)**:PD 解耦是 stage-level 分离 (prefill vs decode),本文研究的 AFD 是 component-level 进一步分离 (decode 内部 Attention vs FFN);两者可叠加形成 PD-AFD 联合架构
- **vs MoonCake / SpeedyLLM 等 AFD 实现 (Wang et al. 2025, Zhu et al. 2025)**:这些工作给出 AFD 的系统实现并经验地搜 $r$,本文给出他们缺的理论支撑
- **vs Sarathi (Agrawal et al. 2024)**:Sarathi 用 chunked prefill 让 batch 更稳定,本文反过来假设 batch 稳定再优化 component ratio,两条优化轴互补
- **vs VirtualFlow / FlexFlow 类流水线优化**:他们做的是给定模型分图,本文做的是给定模型架构选 deployment ratio,目标和粒度都不一样

## 评分
- 新颖性: ⭐⭐⭐⭐ AFD 领域首个有闭式解的理论框架,几何分布 + 三 regime 的拆解角度新颖
- 实验充分度: ⭐⭐⭐ 模拟器跨多工作负载验证 <10% 误差很扎实,但缺真实硬件部署验证略减分
- 写作质量: ⭐⭐⭐⭐⭐ 从工作负载建模到 Lemma 4.1、Prop 4.3、Theorem 4.4 的推导链条非常清晰,Practical Recipe 部分让工程师能直接落地
- 价值: ⭐⭐⭐⭐ 给 AFD 配比这个工程经验问题提供了 principled 公式,工业部署可省大量 trial-and-error;若有真机验证会更强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] DOT-MoE: 用可微 optimal transport 把 dense LLM 转成 MoE](dot-moe_differentiable_optimal_transport_for_moefication.md)
- [\[ICML 2026\] GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving.md)
- [\[ICML 2026\] OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)
- [\[ICML 2026\] Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)
- [\[NeurIPS 2025\] UMoE: Unifying Attention and FFN with Shared Experts](../../NeurIPS2025/llm_efficiency/umoe_unifying_attention_and_ffn_with_shared_experts.md)

</div>

<!-- RELATED:END -->
