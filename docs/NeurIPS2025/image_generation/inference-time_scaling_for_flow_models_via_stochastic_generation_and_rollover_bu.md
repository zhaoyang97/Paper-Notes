---
title: >-
  [论文解读] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing
description: >-
  [NeurIPS2025][图像生成][Flow Models] 提出针对 Flow 模型的推理时扩展方法：通过 ODE→SDE 转换引入随机性以启用粒子采样，利用线性→VP 插值变换扩大搜索空间，并设计 Rollover Budget Forcing (RBF) 策略自适应分配计算预算，在组合文本生成图像和数量感知生成任务上显著超越所有现有方法。
tags:
  - "NeurIPS2025"
  - "图像生成"
  - "Flow Models"
  - "推理时扩展"
  - "粒子采样"
  - "SDE转换"
  - "插值变换"
  - "Rollover Budget Forcing"
  - "FLUX"
---

# Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing

**会议**: NeurIPS2025  
**arXiv**: [2503.19385](https://arxiv.org/abs/2503.19385)  
**代码**: [flow-inference-time-scaling](https://flow-inference-time-scaling.github.io/)  
**领域**: 图像生成 / 推理时扩展  
**关键词**: Flow Models, 推理时扩展, 粒子采样, SDE转换, 插值变换, Rollover Budget Forcing, FLUX

## 一句话总结

提出针对 Flow 模型的推理时扩展方法：通过 ODE→SDE 转换引入随机性以启用粒子采样，利用线性→VP 插值变换扩大搜索空间，并设计 Rollover Budget Forcing (RBF) 策略自适应分配计算预算，在组合文本生成图像和数量感知生成任务上显著超越所有现有方法。

## 研究背景与动机

- **推理时扩展 (Inference-Time Scaling)**: 近年来 LLM 领域（OpenAI o1、DeepSeek R1）通过在推理阶段投入更多计算来提升输出质量，这一思路也被引入扩散模型
- **扩散模型的粒子采样**: 对扩散模型而言，由于中间去噪步骤天然具有随机性，粒子采样方法（SVDD、CoDe、SMC）可在去噪过程中维护多个候选样本并筛选高奖励粒子，比简单的 Best-of-N 高效得多
- **Flow 模型的困境**: Flow 模型（如 FLUX）凭借更快的生成速度和更高质量的输出成为主流，但其基于 ODE 的确定性生成过程使得给定同一起始噪声只能产生唯一样本，**粒子采样方法无法直接应用**
- **现有工作不足**: 此前唯一针对 Flow 模型的推理时扩展方法 Search over Paths (SoP) 仅通过前向核采样粒子，未探索修改反向核的可能性，限制了扩展效率

## 核心问题

给定预训练的 Flow 模型，如何在不额外训练的情况下，通过推理时计算扩展生成与用户偏好/奖励函数高度对齐的样本？

形式化目标：

$$p_0^* = \arg\max_q \ \mathbb{E}_{\mathbf{x}_0 \sim q}[r(\mathbf{x}_0)] - \beta \mathcal{D}_{\text{KL}}[q \| p_0]$$

即最大化期望奖励的同时，通过 KL 散度正则化防止偏离原始分布太远。

## 方法详解

### 1. 推理时 SDE 转换 (Inference-Time SDE Conversion)

Flow 模型通过求解 Probability Flow ODE 生成样本：

$$\mathrm{d}\mathbf{x}_t = u_t(\mathbf{x}_t)\mathrm{d}t$$

该过程是完全确定性的——同一噪声只产生一个样本，所有粒子坍缩到同一点。

**关键转换**: 将确定性 ODE 转换为具有相同边际分布的随机 SDE：

$$\mathrm{d}\mathbf{x}_t = \mathbf{f}_t(\mathbf{x}_t)\mathrm{d}t + g_t\mathrm{d}\mathbf{w}$$

其中：
- 漂移系数 $\mathbf{f}_t(\mathbf{x}_t) = u_t(\mathbf{x}_t) - \frac{g_t^2}{2}\nabla\log p_t(\mathbf{x}_t)$
- $g_t$ 为可自由选择的扩散系数（本文设 $g_t = t^2$，缩放因子3）
- 分数函数可通过预训练速度场计算：$\nabla\log p_t(\mathbf{x}_t) = \frac{1}{\sigma_t}\frac{\alpha_t u_t(\mathbf{x}_t) - \dot{\alpha}_t \mathbf{x}_t}{\dot{\alpha}_t \sigma_t - \alpha_t \dot{\sigma}_t}$

转换后的提议分布为：

$$p_\theta(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_t - \mathbf{f}_t(\mathbf{x}_t)\Delta t,\ g_t^2 \Delta t\ \mathbf{I})$$

这使粒子采样成为可能——从同一 $\mathbf{x}_t$ 可采样出多个不同的 $\mathbf{x}_{t-\Delta t}$。

### 2. 推理时插值变换 (Inference-Time Interpolant Conversion)

Flow 模型使用**线性插值** $(\alpha_t = 1-t, \sigma_t = t)$，而扩散模型常用 **Variance Preserving (VP) 插值**。两种插值通过 scale-time 变换相互转换：

$$\bar{\mathbf{x}}_s = c_s \mathbf{x}_{t_s}, \quad t_s = \rho^{-1}(\bar{\rho}(s)), \quad c_s = \bar{\sigma}_s / \sigma_{t_s}$$

在新插值下的速度场为：

$$\bar{u}_s(\bar{\mathbf{x}}_s) = \frac{\dot{c}_s}{c_s}\bar{\mathbf{x}}_s + c_s \dot{t}_s u_{t_s}\left(\frac{\bar{\mathbf{x}}_s}{c_s}\right)$$

**VP-SDE 的优势**:
- VP 插值在各时间步维持更低的 log-SNR，意味着每步的噪声成分更大，产生更多样化的样本
- 插值变换协同结合了 **时间步转换**（在更低 SNR 处采样）和 **扩散系数缩放**（增大方差），单独使用任一机制效果不佳，组合使用则在不损害质量的前提下显著提高多样性

### 3. Rollover Budget Forcing (RBF)

此前方法（SVDD、CoDe）在所有去噪步均匀分配计算预算（每步固定 NFE 数量），但实验发现不同步骤需要的计算量差异很大。

**RBF 策略**:
1. 将总 NFE 预算均匀分配为每步配额 $Q$
2. 在每步中，一旦发现奖励更高的粒子 $\mathbf{x}_{t-\Delta t}$（优于当前 $\mathbf{x}_t$），**立即跳到下一步**
3. 剩余的 NFE 配额**滚存到后续步骤**
4. 若配额耗尽仍未找到更好样本，选择当前粒子集中期望未来奖励最高的进入下一步

这种自适应分配策略避免了在"容易改进"的步骤浪费预算，将计算集中在需要更多探索的步骤上。

### 4. 未来奖励估计

粒子选择依据是**期望未来奖励**，通过 Tweedie 公式的后验均值近似：

$$v(\mathbf{x}_t) \approx r(\mathbf{x}_{0|t}), \quad \mathbf{x}_{0|t} \coloneq \mathbb{E}_{\mathbf{x}_0 \sim p_\theta(\mathbf{x}_0|\mathbf{x}_t)}[\mathbf{x}_0]$$

Flow 模型（尤其是经过 rectification 微调的）在中间步骤的后验均值更清晰，使奖励估计更精准，这是 Flow 模型相较扩散模型在推理时扩展上的独特优势。

## 实验关键数据

### 实验设置
- 预训练模型：**FLUX**
- 总 NFE 预算：500，去噪步数 10（每步50 NFE）
- 对比方法：BoN、SoP、SMC (DAS)、CoDe、SVDD

### 组合文本生成图像 (GenAI-Bench, 121 prompts)
- **给定奖励** (VQAScore): VP-SDE + RBF 达到最高，显著超越所有 Linear-ODE 方法
- **未见奖励** (InstructBLIP): VP-SDE 同样最优，证明泛化能力
- **图像质量** (Aesthetic Score): VP-SDE 与基础 FLUX 相当，未损害生成质量
- 性能提升路径：Linear-ODE → Linear-SDE → VP-SDE，所有粒子采样方法均获一致提升

### 数量感知图像生成 (T2I-CompBench++, 100 prompts)
- VP-SDE + RBF 达到最高准确率，**相比基础 FLUX 提升 4~6 倍**
- Linear-SDE 已超越所有 Linear-ODE 方法（BoN 和 SoP），VP 进一步提升

### 消融实验 (LPIPS-MPD 多样性度量)

| 方法 | LPIPS-MPD ↑ | VQAScore ↑ | Inst. BLIP ↑ |
|------|------------|------------|-------------|
| Linear-ODE | – | 0.788 | 0.789 |
| Linear-SDE | 0.158 | 0.900 | 0.813 |
| + 自适应时间步 | 0.270 | 0.908 | 0.813 |
| + 自适应扩散系数 | 0.429 | 0.702 | 0.571 |
| VP-SDE | **0.509** | **0.925** | **0.843** |

- 单独增大扩散系数虽提高多样性但严重损害质量
- VP-SDE 通过时间步转换 + 扩散系数缩放的协同作用，同时实现最高多样性和最高奖励

## 理论基础：SDE 转换的正确性

论文在附录中给出了完整的理论证明。核心命题：

**Proposition 1**: 对于线性随机过程 $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \mathbf{x}_1$ 及其 Probability-Flow ODE $\mathrm{d}\mathbf{x}_t = u_t(\mathbf{x}_t)\mathrm{d}t$，以下前向和反向 SDE（任意扩散系数 $g_t \geq 0$）共享相同的边际密度：

- 前向 SDE: $\mathrm{d}\mathbf{x}_t = [u_t(\mathbf{x}_t) + \frac{g_t^2}{2}\nabla\log p_t(\mathbf{x}_t)]\mathrm{d}t + g_t\mathrm{d}\mathbf{w}$
- 反向 SDE: $\mathrm{d}\mathbf{x}_t = [u_t(\mathbf{x}_t) - \frac{g_t^2}{2}\nabla\log p_t(\mathbf{x}_t)]\mathrm{d}t + g_t\mathrm{d}\mathbf{w}$

证明关键步骤是将 Fokker-Planck 方程与连续性方程对齐，要求 $p_t(\mathbf{x}_t)(\mathbf{f}_t(\mathbf{x}_t) - u_t(\mathbf{x}_t)) = \frac{g_t^2}{2}\nabla p_t(\mathbf{x}_t)$，从而得到漂移系数的修正项。

**Corollary 1**: 当扩散系数选择为 $g_t = \sqrt{2(\sigma_t\dot{\sigma}_t - \sigma_t^2 \dot{\alpha}_t/\alpha_t)}$ 时，前向 SDE 中的分数函数项消失，简化为 $\mathrm{d}\mathbf{x}_t = \frac{\dot{\alpha}_t}{\alpha_t}\mathbf{x}_t\mathrm{d}t + g_t\mathrm{d}\mathbf{w}$。这意味着存在特殊的扩散系数使前向过程不依赖分数函数。

## 搜索算法实现细节

论文附录详细给出了各算法的具体参数配置：

| 算法 | 批大小 N | 粒子数 K | 间隔 L | 总 NFE |
|------|---------|---------|--------|--------|
| **BoN** | 50 | — | — | 500 |
| **SoP** | 2 | 5 | — | 500 |
| **SMC (DAS)** | 50 | — | — | 500 |
| **CoDe** | 2 | 25 | 2 | 500 |
| **SVDD** | 2 | 25 | — | 500 |
| **RBF (Ours)** | 2 | 自适应 | — | 500 |

**RBF 伪代码核心逻辑**:
1. 初始化：采样 $\bar{\mathbf{x}}_1 \sim \mathcal{N}(0, \mathbf{I})$，计算初始奖励 $r^* \leftarrow r(\bar{\mathbf{x}}_{0|1})$
2. 对每个去噪步 $i$：分配配额 $q \leftarrow Q^{(i)}$
3. 逐个采样粒子 $\bar{\mathbf{x}}_{s-\Delta s}^{(j)}$，若 $r(\bar{\mathbf{x}}_{0|s-\Delta s}^{(j)}) > r^*$，则更新当前最优，剩余配额 $Q^{(i)} - j$ 滚存到 $Q^{(i+1)}$，立即 break
4. 若配额耗尽仍无更优粒子，选择当前集中奖励最高者

**自适应时间调度**: VP-SDE 采样使用非均匀时间调度 $t_{\text{new}} = \sqrt{1-(1-t)^2}$，在初期（方差大）取更小步长以充分探索，后期逐渐增大步长。10 步去噪下该设置表现良好，得益于 Flow 模型的 few-step 生成能力。

**NFE 分析**: 论文分析了在各时间步获得更高奖励样本所需的 NFE 数量，发现其方差很大——某些步可能只需 1-2 次即找到更优粒子，而其他步可能需要远超均匀分配的预算。均匀分配策略会在"容易"的步浪费预算，在"困难"的步计算不足。

## 美学图像生成实验 (附录)

当奖励函数**可微**时（如 Aesthetic Score），RBF 可与梯度方法 DPS 结合产生协同效果：

| 方法 | Aesthetic Score† ↑ | ImageReward (held-out) ↑ |
|------|-------------------|------------------------|
| FLUX (基线) | 5.795 | 0.991 |
| DPS | 6.438 | 0.605 |
| SVDD + DPS | 6.887 | 1.077 |
| **RBF + DPS** | **7.170** | **1.152** |

- 单独使用 DPS 虽提升美学分数，但 held-out 奖励 (ImageReward) 反而下降，说明存在奖励过优化
- SVDD + DPS 和 RBF + DPS 均在两个指标上实现提升，RBF + DPS 达到最优
- 这验证了粒子采样与梯度方法的互补性：梯度提供局部优化方向，粒子采样提供全局探索

## 亮点

1. **首次实现 Flow 模型的粒子采样**: 通过推理时 ODE→SDE 转换，无需重训练即可将扩散模型的高效粒子采样方法引入 Flow 模型
2. **VP 插值变换的理论洞察**: 从 log-SNR 角度分析了 VP 插值如何通过时间步转换和扩散系数缩放的协同作用扩大搜索空间，避免了单纯增大噪声导致的质量下降
3. **Rollover Budget Forcing 策略简洁高效**: 无需额外超参调节，自适应分配计算预算，在所有粒子采样方法之上进一步提升性能
4. **模型无关性**: 方法不修改预训练模型，可直接应用于任何 Flow 模型（如 FLUX），且可与梯度方法（如 DPS）结合产生协同效果
5. **Flow 模型的独特优势**: 由于 rectified flow 的轨迹更直，中间步后验均值更清晰，使得 Flow 模型在推理时扩展中的奖励估计比扩散模型更精准
6. **完整的理论保证**: 通过 Fokker-Planck 方程严格证明了 SDE 转换保持边际密度不变，方法的正确性有坚实理论基础

## 局限与展望

1. **推理开销**: 引入额外的推理时计算，当基础模型预测计算密集时可能成为瓶颈
2. **安全风险**: 预训练模型可能在未经审核的数据集上训练，推理时扩展有被恶意利用生成不当内容的风险
3. **扩散系数选择**: $g_t = t^2$ 的选择缺乏理论最优性保证，系统性探索时间步调度和扩散系数缩放是未来方向
4. **奖励函数依赖**: 方法效果取决于奖励函数质量，非可微奖励限制了梯度方法的集成
5. **视频生成验证**: 虽然 Flow 模型广泛用于视频生成（如 Goku），但实验仅验证了图像任务
6. **奖励过优化风险**: 单独使用 DPS 时 held-out 指标下降，说明过度优化给定奖励可能损害泛化性

## Training / Inference 细节

- **无需训练**: 本方法完全在推理时操作，不修改预训练模型权重。SDE 转换和插值变换仅需利用已有的速度场 $u_t$
- **推理流程**: 从标准高斯采样初始噪声 → 按 VP-SDE 执行 10 步去噪 → 每步通过 Tweedie 公式估计 $\mathbf{x}_{0|t}$ → 计算奖励 $r(\mathbf{x}_{0|t})$ → RBF 策略决定是否继续采样或跳到下一步
- **计算开销**: 总 NFE 固定为 500（与 BoN 50 条样本等价），但因粒子采样在中间步即可筛选，效率远高于 BoN
- **奖励函数**: 可对接任意奖励函数（VQAScore、RSS、Aesthetic Score 等），可微奖励可进一步结合梯度方法（DPS）获得协同提升
- **扩散系数设定**: $g_t = t^2$ 乘以缩放因子 3，在早期步（$t$ 大）注入更多噪声以增加探索，后期步噪声自然减小以保证质量
- **自适应时间调度**: VP-SDE 使用 $t_{\text{new}} = \sqrt{1-(1-t)^2}$ 非均匀调度，初期小步探索、后期大步收敛
- **兼容性**: 可直接应用于 FLUX、Stable Diffusion 3 等基于 Flow Matching 的模型，也可叠加在 fine-tuned 模型之上进一步提升

## 与相关工作的对比

| 方法 | 类型 | 模型类型 | 是否需要随机性 | 预算分配 | 关键特点 |
|------|------|---------|-------------|---------|--------|
| **BoN** | Baseline | 通用 | 否 | 全部前置 | 独立采样 N 条，选最优 |
| **SoP** (Ma et al.) | 粒子采样 | Flow | 前向核注入 | 均匀 | 唯一此前针对 Flow 模型的方法，仅修改前向核 |
| **SVDD** (Li et al.) | 粒子采样 | Diffusion | 天然具有 | 均匀 | 每步选奖励最高粒子 |
| **CoDe** (Singh et al.) | 粒子采样 | Diffusion | 天然具有 | 均匀/间隔选择 | 间隔若干步才做粒子选择 |
| **DAS/SMC** (Kim et al.) | SMC | Diffusion | 天然具有 | 均匀 | 按重要性权重多项式采样 |
| **本文 (VP-SDE + RBF)** | 粒子采样 | **Flow** | **推理时引入** | **自适应 (Rollover)** | ODE→SDE + 线性→VP 插值 + 自适应预算 |

**关键差异**:
- 相比 SoP：本文探索了**反向核的修改**（SDE 转换 + 插值变换），搜索空间更大，且 RBF 提供自适应预算分配
- 相比 SVDD/CoDe/DAS：这些方法依赖扩散模型天然的随机性，无法直接用于 Flow 模型；本文通过推理时 SDE 转换桥接了这一鸿沟
- 相比 BoN：在相同总 NFE 预算下，粒子采样在中间步即筛选低质量样本，避免了完整生成后再淘汰的浪费

## 启发与关联

1. **ODE↔SDE 等价性的实用价值**: 概率流 ODE 与相应 SDE 共享边际分布这一理论性质，在本文中被巧妙地用于实际目的（启用粒子采样），展示了理论工具的工程应用价值
2. **推理时计算扩展的通用范式**: 从 LLM 的 test-time compute（如 o1 的 chain-of-thought）到扩散模型的粒子采样，再到本文的 Flow 模型扩展，"推理时投入更多计算换取更好结果" 正在成为跨模态的通用范式
3. **VP 插值的搜索空间优势**: VP 插值天然在早期步维持更低 SNR，这不仅对生成多样性有利，对其他需要探索的任务（如 RLHF 采样、多样性生成）也有启示
4. **Rollover Budget 策略的普适性**: "发现好结果就提前跳到下一步，剩余预算留给后续" 这一策略不限于生成模型，可推广到任何分步决策 + 预算受限的场景（如 beam search 的动态宽度调整）
5. **Flow 模型的独特优势被定量验证**: rectified flow 的较直轨迹使中间步预测更准确，这为 Flow 模型在需要中间步质量评估的任务中提供了理论支撑
6. **与视频生成的关联**: Goku 等视频 Flow 模型理论上可直接应用本方法，但视频的奖励函数设计（时序一致性、运动质量）是关键挑战

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次系统研究 Flow 模型的粒子采样，三个组件（SDE 转换、VP 插值、RBF）各有独立贡献
- 实验充分度: ⭐⭐⭐⭐ — 两个主要任务 + 消融实验 + 附录中的美学生成和 Flow vs Diffusion 对比，但缺少视频实验
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导清晰，从问题动机到方法再到实验逻辑链完整，图表质量高
- 价值: ⭐⭐⭐⭐ — 为 Flow 模型生态引入了推理时扩展能力，实用性强，但推理开销限制了大规模部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](../../ICML2025/image_generation/performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [\[CVPR 2026\] Rethinking Prompt Design for Inference-time Scaling in Text-to-Visual Generation](../../CVPR2026/image_generation/rethinking_prompt_design_for_inference-time_scaling_in_text-to-visual_generation.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[NeurIPS 2025\] Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models](psi-sampler_initial_particle_sampling_for_smc-based_inference-time_reward_alignm.md)
- [\[NeurIPS 2025\] BoltzNCE: Learning Likelihoods for Boltzmann Generation with Stochastic Interpolants](boltznce_learning_likelihoods_for_boltzmann_generation_with_stochastic_interpola.md)

</div>

<!-- RELATED:END -->
