---
title: >-
  [论文解读] ERTACache: Error Rectification and Timesteps Adjustment for Efficient Diffusion
description: >-
  [ICLR 2026][模型压缩][扩散模型] ERTACache 把"缓存加速带来的画质下降"形式化拆成**特征偏移误差**和**步长放大误差**两类，再用"离线策略标定 + 轨迹感知步长调整 + 闭式残差线性化校正"三件套同时压制这两类误差，在视频/图像扩散模型上做到 2× 以上加速且画质几乎无损。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "扩散模型"
  - "特征缓存"
  - "推理加速"
  - "视频生成"
  - "误差校正"
  - "Flow Matching"
---

# ERTACache: Error Rectification and Timesteps Adjustment for Efficient Diffusion

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=InvyBiYcK5](https://openreview.net/forum?id=InvyBiYcK5)  
**代码**: 待确认  
**领域**: 扩散模型加速 / 特征缓存  
**关键词**: 扩散模型, 特征缓存, 推理加速, 视频生成, 误差校正, Flow Matching  

## 一句话总结
ERTACache 把"缓存加速带来的画质下降"形式化拆成**特征偏移误差**和**步长放大误差**两类，再用"离线策略标定 + 轨迹感知步长调整 + 闭式残差线性化校正"三件套同时压制这两类误差，在视频/图像扩散模型上做到 2× 以上加速且画质几乎无损。

## 研究背景与动机

**领域现状**：扩散模型（尤其是基于 DiT 的视频生成）推理极慢——在 A800 上用 Wan2.1 生成一段 81 帧 480p 视频要约 200 秒，根因是每个去噪步都要前向一遍庞大的 Transformer。在诸多加速手段中，**特征缓存（feature caching）** 最实用：它利用相邻时间步特征高度相似的特性，把某些步的中间输出存下来跨步复用，从而免去整次前向，且**无需重新训练**、易迁移到各种扩散骨干。

**现有痛点**：现有缓存方法各有短板。缓存内部 Transformer 状态（AdaCache、BAC、DiTFastAttn）的方法显存随粒度二次增长，长视频容易 OOM；以 TeaCache 为代表的"在线启发式"方法靠输入相关的阈值动态预测该不该复用，但**预测的复用质量与真实重构误差之间存在系统性偏差**——论文实测（图2a）发现 TeaCache 预测的 $\ell_1$ 距离在后期时间步严重偏离真值，导致复用决策不可靠。

**核心矛盾**：缓存越激进（复用越多步）→ 加速越大，但误差会沿 ODE 轨迹**逐步累积放大** → 画质崩坏。如何在激进复用下仍把累积误差压住，是缓存加速的核心难题。

**核心 idea**：**先把误差讲清楚，再对症下药**。论文先做了一个关键观察——尽管扩散轨迹依赖输入，但**缓存复用模式和误差在不同 prompt 间高度一致**（图2a 蓝线在各步变化很小），这意味着一个通用的、离线优化好的缓存策略足以逼近最优，不必每次在线预测。基于此，作者把缓存误差形式化分解为两类，并设计三个互补组件分别针对性校正。

## 方法详解

### 整体框架
ERTACache 在 Flow-Matching 框架下工作：前向是线性插值 $x_t=(1-t)x_0+t\cdot x_T$，采样时沿学到的 ODE 用 Euler 法积分 $x_{i-1}=x_i+\Delta t_i\cdot v_\theta(x_i,t)$。缓存复用的是**残差** $\tilde r=v_\theta(x_{i+1},t)-x_{i+1}$：若第 $i$ 步被跳过，则用 $\tilde v_i=x_i+\tilde r$ 重建输出，省去一次前向。整个流程分三段：**(a) 离线策略标定**用一小批校准样本搜出"哪些步可复用"的缓存集合 $S$；**(b) 时间步调整**在复用步上动态缩放积分步长以纠正轨迹漂移；**(c) 推理时误差校正**用预先算好的闭式系数补偿缓存引入的加性误差。

```mermaid
flowchart LR
    A[离线全量推理<br/>记录真值残差 r_gt] --> B[阈值搜索<br/>ℓ1rel < λ?]
    B --> C[得到可复用步集合 S]
    C --> D[推理: t∈S 复用 r̃<br/>否则重算并刷新缓存]
    D --> E[轨迹感知步长调整<br/>Δt_i = Δt_c · φ_i]
    E --> F[闭式误差校正<br/>v̂_i = ṽ_i - ε_i]
    F --> G[高保真加速采样输出]
```

### 关键设计

**1. 缓存误差的形式化分解：把"画质下降"拆成两个可对症的来源。** 这是全文的理论地基。设缓存输出 $\tilde v_i=v_i+\varepsilon_i$，其中 $\varepsilon_i$ 是缓存近似引入的加性误差。把 $\tilde v_{i+1}$ 代入 Euler 更新，单步轨迹偏离为 $\delta_i=\tilde x_i-x_i=\Delta t_{i+1}\cdot\varepsilon_{i+1}$。若从第 $i$ 步起连续缓存 $m$ 步，偏离会沿轨迹累积成 $\delta_{i-m}=\sum_{k=0}^{m-1}\Delta t_{i-k}\cdot\varepsilon_{i-k}$。这个式子清楚地点出两个误差源：$\varepsilon_{i-k}$ 是**特征偏移误差（Feature Shift Error）**——缓存特征与真值计算之间的差异本身；$\Delta t_{i-k}$ 这个权重则把误差**逐步放大**，构成**步长放大误差（Step Amplification Error）**。后面三个组件正是分别针对"减小 $\varepsilon$"和"控制 $\Delta t$ 放大"来设计的。

**2. 离线策略标定：用残差画像把"在线猜"换成"离线搜"。** 既然误差模式跨 prompt 一致，就没必要在线启发式预测。作者先在小校准集上跑全量推理，记录所有步的真值残差 $r_{gt}(x_i,t)$；然后对一系列候选阈值 $\lambda$，用相对 $\ell_1$ 误差 $\ell_{1rel}(x_i,t)=\frac{\|\tilde r_{cali}-r_{cali}(x_i,t)\|_1}{\|r_{gt}(x_i,t)\|_1}$ 判断缓存残差是否足够接近真值——若 $\ell_{1rel}<\lambda$ 则复用，否则重算并刷新缓存。扫不同 $\lambda$ 就得到一个全局有效的可复用步集合 $S=\{s_0,\dots,s_c\}$。$\lambda$ 越小复用越少、细节越保真，越大越快但可能掉质量，需要权衡。这一步直接降低了无谓复用造成的特征偏移误差。

**3. 轨迹感知时间步调整：缓存复用后顺手把积分步长改对。** 朴素复用默认严格沿原 ODE 轨迹走，但固定步长下复用会引起明显的轨迹漂移（图2b）。作者引入校正系数 $\phi_i\in[0,1]$ 动态缩放步长：非复用步用 $\Delta t_i=\Delta t_c$，复用步用 $\Delta t_i=\Delta t_c\cdot\phi_i$，其中 $\phi_i=\mathrm{clip}\!\left(1-\frac{\|\tilde v_i-v_i\|_1}{\|v_i-v_{i+1}\|_1},0,1\right)$——即缓存输出越偏离真值，就越压缩这一步的积分步长，减弱误差被放大的程度。每步更新后再按 $\Delta t_c=\frac{1-\sum_{j=0}^{i}\Delta t_j}{1-i/T}$ 重新分配剩余时间预算，保证整条轨迹仍能在 $[0,1]$ 内对齐目标采样路径。这一项直接打击步长放大误差。

**4. 闭式残差线性化校正：给每个缓存步补一个预先算好的误差项。** 加性误差 $\varepsilon_i=\tilde v_i-v_i$ 因 prompt 结构难以直接预测，作者用一个轻量线性化模型近似 $\varepsilon_i=\sigma(K_i\cdot\tilde v_i+B_i)$（$\sigma$ 为 sigmoid），在 MSE 损失下拟合 $K_i,B_i$。关键技巧是对 sigmoid 做一阶 Taylor 展开 $\sigma(K_i\tilde v_i+B_i)\approx\frac14(K_i\tilde v_i+B_i)+\frac12$，从而把最小二乘问题求出**闭式解** $K_i\approx 4\cdot\frac{S_{v_i\varepsilon_i}}{S_{v_iv_i}}$、$B_i\approx 4(\bar\varepsilon_i-\frac12)-4K_i\bar v_i$（其中 $S$ 为对应方差/协方差项）。这些系数可在小数据集上**预先算好**，推理时直接用来对缓存输出做误差校正，额外开销 <0.5%。

## 实验关键数据

### 主实验表格（视频生成，VBench）

| 模型 / 方法 | Speedup↑ | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|---|
| **Open-Sora 1.2** (T=30) | 1× | 79.22% | - | - | - |
| TeaCache-slow | 1.55× | 79.28% | 0.1316 | 0.8415 | 23.62 |
| **ERTACache-slow** (λ=0.1) | 1.55× | **79.36%** | **0.1006** | **0.8706** | **25.45** |
| TeaCache-fast | 2.25× | 78.48% | 0.2511 | 0.7477 | 19.10 |
| **ERTACache-fast** (λ=0.18) | **2.47×** | **78.64%** | **0.1659** | **0.8170** | **22.34** |
| **CogVideoX** (T=50) | 1× | 80.18% | - | - | - |
| TeaCache | 2.92× | 79.00% | 0.2057 | 0.7614 | 20.97 |
| **ERTACache-fast** (λ=0.3) | **2.93×** | 78.79% | **0.1012** | **0.8702** | **26.44** |
| **Wan2.1-1.3B** (T=50) | 1× | 81.30% | - | - | - |
| TeaCache | 2.00× | 76.04% | 0.2913 | 0.5685 | 16.17 |
| ProfilingDiT | 2.01× | 76.15% | 0.1256 | 0.7899 | 22.02 |
| **ERTACache** (λ=0.08) | **2.17×** | **80.73%** | **0.1095** | **0.8200** | **23.77** |

图像生成（Flux-dev 1.0，T=30）：ERTACache（λ=0.6）1.86× 加速，CLIP 0.9534 / LPIPS 0.3029 / SSIM 0.8962 / PSNR 20.51，全面优于同等加速比的 TeaCache（0.9065 / 0.4427 / 0.7445 / 16.48）。

### 消融实验表格（逐组件累加）

| Wan2.1-1.3B | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|
| Uniform Cache（均匀复用） | 79.35% | 0.5041 | 0.4058 | 13.76 |
| + Offline Policy | 80.59% | 0.1477 | 0.7738 | 22.09 |
| + Time Adjustment | **80.89%** | 0.1267 | 0.7988 | 22.94 |
| + Error Rectification | 80.73% | **0.1095** | **0.8200** | **23.77** |

### 关键发现
- **三组件互补、层层递进**：离线策略奠定基础恢复（LPIPS 从 0.50 骤降到 0.15），时间步调整优化结构/时序一致性（Flux 上 PSNR/SSIM 各 +0.95/+0.023），误差校正最大化细节保留（PSNR 再 +0.83）。
- **离线策略相比均匀缓存在 Wan2.1 上 VBench 提升 1.24%**（统计显著），说明"搜对该复用的步"本身就能大幅减少信息损失。
- **Wan2.1 上最突出**：在与 TeaCache 相近甚至更高加速比下，VBench 几乎无损（80.73% vs 全量 81.30%），而 TeaCache 掉到 76.04%。
- 误差校正等组件可离线预计算，推理额外开销 **<0.5%**。

## 亮点与洞察
- **"先解释再解决"的范式**：先把缓存误差严格分解成可解释的两类，再让每个组件对应一类误差，方法论清晰、可解释性强，避免了纯启发式调阈值的玄学。
- **离线 > 在线的关键洞察**：发现缓存误差跨 prompt 高度一致，从而把"每次在线预测"换成"一次离线标定全局复用"，既稳又省。
- **闭式解的工程美感**：用 sigmoid 一阶 Taylor 展开把误差拟合化简成最小二乘闭式解，系数可预计算，几乎零推理开销就拿到误差校正。
- **泛化性强**：在 Open-Sora、CogVideoX、Wan2.1、Flux-dev 四种骨干、视频+图像两类任务上都稳定生效。

## 局限与展望
- **依赖离线标定集**：策略 $S$ 和校正系数都在小校准集上搜/拟合，若部署分布与标定分布差异较大，"跨 prompt 一致性"假设可能松动，泛化边界尚未充分探讨。
- **线性化近似的精度上限**：误差用 sigmoid 一阶 Taylor 线性化建模，对高度非线性的误差结构可能拟合不足；论文未深入分析近似误差本身的影响。
- **超参 λ 仍需调**：不同模型/速度档位要选不同 $\lambda$（0.08~0.6 跨度很大），自动选 λ 的机制缺失。
- 与 token 级方法（FastCache、TokenCache）正交互补，论文指出可叠加但未给出组合实验。

## 相关工作与启发
- **vs TeaCache**：TeaCache 在线启发式预测复用，预测误差后期发散；ERTACache 改为离线标定 + 显式误差建模，更可靠。
- **vs AdaCache/BAC/DiTFastAttn**：这类缓存内部状态/注意力图的方法显存二次增长易 OOM；ERTACache 只缓存残差，显存友好。
- **vs LazyDiT/ICC**：LazyDiT 加额外训练损失正则化缓存特征（有训练开销），ICC 用低秩校准缓存特征；ERTACache 显式建模残差误差、免重训、且兼顾精度与显存。
- **启发**：把"加速带来的误差"形式化分解再对症设计，比单一启发式更稳；离线标定 + 闭式校正的思路可迁移到量化、蒸馏等其他无训练加速场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 误差两分解 + 闭式残差线性化校正的组合较新颖，把缓存加速从"调阈值启发式"推进到"可解释的对症校正"。
- **实验充分度**: ⭐⭐⭐⭐ — 四骨干、视频+图像、多 baseline、逐组件消融都覆盖，主实验扎实；但缺自动选 λ、分布偏移鲁棒性等深入分析。
- **写作质量**: ⭐⭐⭐⭐ — 误差分解推导清晰、图示到位，方法叙述逻辑顺畅，公式与动机对应明确。
- **价值**: ⭐⭐⭐⭐ — 无训练即插即用、显存友好、2× 加速近乎无损，对视频扩散落地部署有直接实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Plug-and-Play Fidelity Optimization for Diffusion Transformer Acceleration via Cumulative Error Minimization](plug-and-play_fidelity_optimization_for_diffusion_transformer_acceleration_via_c.md)
- [\[ICML 2026\] FedSDR: Federated Self-Distillation with Rectification](../../ICML2026/model_compression/fedsdr_federated_self-distillation_with_rectification.md)
- [\[CVPR 2026\] Otil: Accelerating Diffusion Model Inference via Communication-Efficient Multi-GPU Parallelism](../../CVPR2026/model_compression/otil_accelerating_diffusion_model_inference_via_communication-efficient_multi-gp.md)
- [\[ICLR 2026\] ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping](es-dllm_efficient_inference_for_diffusion_large_language_models_by_early-skippin.md)
- [\[ICLR 2026\] RAPID$^3$: Tri-Level Reinforced Acceleration Policies for Diffusion Transformer](rapid3_tri-level_reinforced_acceleration_policies_for_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
