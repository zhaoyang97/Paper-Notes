---
title: >-
  [论文解读] Improved Mean Flows: On the Challenges of Fastforward Generative Models
description: >-
  [CVPR 2026][图像生成][一步生成] 论文诊断出 MeanFlow（一步生成框架）的两个病根——训练目标依赖网络自身、CFG 引导尺度训练前被写死——并分别用"把目标重写成网络无关的 v-loss + 用预测的边际速度当 JVP 输入"和"把引导尺度当作可变条件 + 多 token in-context 条件注入"对症下药，得到的 iMF 在 ImageNet 256×256 上单次函数评估（1-NFE）从零训练拿到 1.72 FID，比原 MeanFlow 相对提升约 50%，逼近多步方法且全程不蒸馏。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "一步生成"
  - "MeanFlow"
  - "flow matching"
  - "classifier-free guidance"
  - "1-NFE"
---

# Improved Mean Flows: On the Challenges of Fastforward Generative Models

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Geng_Improved_Mean_Flows_On_the_Challenges_of_Fastforward_Generative_Models_CVPR_2026_paper.html)  
**代码**: 复用原 MeanFlow 公开代码（论文未单列新仓库）  
**领域**: 图像生成  
**关键词**: 一步生成, MeanFlow, flow matching, classifier-free guidance, 1-NFE

## 一句话总结
论文诊断出 MeanFlow（一步生成框架）的两个病根——训练目标依赖网络自身、CFG 引导尺度训练前被写死——并分别用"把目标重写成网络无关的 v-loss + 用预测的边际速度当 JVP 输入"和"把引导尺度当作可变条件 + 多 token in-context 条件注入"对症下药，得到的 iMF 在 ImageNet 256×256 上单次函数评估（1-NFE）从零训练拿到 1.72 FID，比原 MeanFlow 相对提升约 50%，逼近多步方法且全程不蒸馏。

## 研究背景与动机
**领域现状**：扩散/流匹配模型把生成看成解一个 ODE，把先验分布映射到数据分布，通常要多步数值求解、多次函数评估（NFE）。近年一批"fastforward（快进）生成模型"把 ODE/SDE 的加速直接写进训练目标，靠跨大时间区间的"一跳"实现极少步甚至一步生成。MeanFlow（MF）是其中代表：不学瞬时速度场 $v$，而学两个时刻之间的**平均速度场** $u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau)\,d\tau$，并用"MeanFlow 恒等式" $u=v-(t-r)\frac{d}{dt}u$ 把不可积的定义变成可训练目标。

**现有痛点**：MF 有两个没解决的麻烦。① **训练目标依赖网络自身**：真值 $u$ 不可得，MF 直接用网络自己的预测 $u_\theta$ 替进目标里（$u_{tgt}=(e-x)-(t-r)\,\mathrm{JVP}(u_\theta;e-x)$），这不是标准回归问题——目标会随网络一起动，训练损失高方差、甚至不下降。② **CFG 尺度训练前写死**：MF 支持 1-NFE 的 classifier-free guidance，但要求引导尺度 $\omega$ 在训练前就固定，推理时没法调；而最优 $\omega$ 其实随模型能力变（更大/训得更久/更多 NFE 的模型偏好更小的 $\omega$），提前冻死必然次优。

**核心矛盾**：fastforward 模型为了"一跳到位"必须在训练时构造跨大区间的 look-ahead 目标，而这个目标里塞进了"网络自己的预测"和"提前定死的超参"，于是**可训练性（目标合法、低方差）**与**这种 look-ahead 构造方式**直接打架。

**本文目标**：(i) 把 MF 的目标改造成一个网络无关的标准回归问题、稳住训练；(ii) 让 CFG 尺度在训练和推理时都能自由取值，同时保住 1-NFE。

**切入角度**：作者发现 MF 的 u-loss 在数学上**完全等价于**一个用 $u_\theta$ 重参数化的 v-loss（瞬时速度损失）。既然如此，不如把回归目标换成网络无关的 $v$，再顺手修掉残留的"非法输入"。

**核心 idea**：把 MeanFlow 看成"被 $u_\theta$ 重参数化的 v-loss"，让回归目标只依赖真值速度、让预测函数只吃噪声样本 $z_t$；再把引导相关超参全部当成可学习条件，用多 token in-context 方式注入。

## 方法详解

### 整体框架
iMF 不改 MeanFlow 的骨架（一步生成、平均速度参数化、ImageNet 隐空间训练），只动**训练目标的写法**和**条件注入方式**，因此本质是一组损失/架构层面的改造，而非新 pipeline。四个改造按逻辑递进：先把 MF 重写成 **v-loss 形式**（拿到网络无关的回归目标 $v$）；这一步会暴露"预测函数还偷吃了 $e-x$"的隐患，于是用 **合法参数化**（把 JVP 的输入从条件速度 $e-x$ 换成网络预测的边际速度 $v_\theta$）让预测函数只依赖 $z_t$；接着把 CFG 从"训练前定死的尺度"改成 **灵活引导条件**（$\omega$ 乃至 CFG 区间都当可学条件）；最后用 **改进的 in-context 条件注入**把众多异构条件（$r,t,c,\Omega$）以多 token 拼接进序列、顺手砍掉笨重的 adaLN-zero。因为是纯目标/机制改造，这里不画 pipeline 图，用公式说清即可。

### 关键设计

**1. MeanFlow 重写成 v-loss：换一个网络无关的回归目标**

MF 的麻烦在于它优化的是 u-loss，而真值 $u$ 不可得、只能拿网络自己的 $u_\theta$ 顶上，目标因此"随网络漂移"。作者把 MeanFlow 恒等式挪个位置写成 $v(z_t)=u(z_t)+(t-r)\frac{d}{dt}u(z_t)$：左边的瞬时速度 $v$ 可以像标准 Flow Matching 那样当**固定的回归目标**，右边的复合函数用 $u_\theta$ 来参数化，记 $V_\theta\triangleq u_\theta(z_t)+(t-r)\,\mathrm{JVP_{sg}}(u_\theta;e-x)$，于是得到一个 Flow-Matching 式的目标 $\mathbb{E}\,\lVert V_\theta-(e-x)\rVert^2$。这个写法被证明与原 MF 目标**完全等价**，却揭示了一个新视角：MeanFlow 就是"被 $u_\theta$ 重参数化的 v-loss"。重点是回归目标 $v$ 不再依赖网络，回到了标准回归问题，训练因此稳得多。

**2. 合法参数化：把 JVP 的输入从条件速度换成预测的边际速度**

v-loss 重写暴露了一个隐患：$V_\theta$ 不只吃 $z_t$，还偷吃了 $e-x$（写作 $V_\theta(z_t,\,e-x)$），从标准回归角度看这是个"非法"预测函数。根子在 Eq.(6) 那步近似——把 JVP 里该用的**边际速度** $v(z_t)=\mathbb{E}[v_c\mid z_t]$ 替成了**条件速度** $v_c=e-x$。作者干脆不替：重新定义 $V_\theta(z_t)\triangleq u_\theta(z_t)+(t-r)\,\mathrm{JVP_{sg}}(u_\theta;v_\theta)$，让 JVP 的切向量也由网络预测的 $v_\theta$ 给出，两个分量都只吃 $z_t$，预测函数就合法了。为零成本实现 $v_\theta$，用边界条件 $v(z_t,t)\equiv u(z_t,t,t)$，直接令 $v_\theta(z_t,t)=u_\theta(z_t,t,t)$（不加任何参数）；也可加一个仅训练时用、推理不用的辅助 v-head 进一步提升。为什么有效：真正唯一的回归目标是低方差的边际速度 $v(z_t)$，而条件速度 $e-x$ 方差很大、作为 JVP 切向量会被雅可比放大，主导损失；换成 $v_\theta$ 这个更低方差的估计后，iMF 的训练损失单调下降、方差远小于原 MF（原 MF 损失高方差、非下降）。stop-gradient 在此是**预测函数的一部分**而非目标的一部分，理论上可去，但实践中保留它能避免对 $\theta$ 的高阶梯度、更好优化。

**3. 灵活引导：把 CFG 尺度（乃至区间）变成可学条件**

原 MF 的 $\omega$ 在训练前定死，推理不可调，而最优 $\omega$ 随训练时长、推理步数、模型规模而变（图 4：更强的设置偏好更小 $\omega$），冻死必然次优。作者把引导尺度**类比时间步 $t,r$ 一样当成条件**喂给网络：$V_\theta(\cdot\mid c,\omega)\triangleq u_\theta(z_t\mid c,\omega)+(t-r)\,\mathrm{JVP_{sg}}$，训练时 $\omega$ 从一个分布里随机采样，推理时任意取值，同一个模型即可在 1-NFE 下扫不同 $\omega$。这套框架还能进一步纳入 CFG 区间 $[t_{min},t_{max}]$（区间外把 $\omega$ 置 1 关掉引导），统一记作条件集 $\Omega=\{\omega,t_{min},t_{max}\}$，每项各有自己的 embedding。这样既解锁了 CFG 的全部灵活性，又不破坏一步采样。

**4. 改进的 in-context 条件注入：多 token 拼接 + 砍掉 adaLN-zero**

iMF 现在有一堆异构条件：两个时间步 $r,t$、类别 $c$、引导集 $\Omega$。常规做法 adaLN-zero 把所有条件 embedding 加在一起，但条件一多、单次相加就"负担过重"、变弱。作者改用 in-context 条件注入——DiT 里它原本被认为逊于 adaLN-zero，但作者发现**只要每个条件用多个 token**就能补上差距：实现中类别用 8 个 token、其余每个条件各 4 个 token，全部作为可学 token 沿序列轴与图像 latent token 拼接，一起过 Transformer 块。一个重要副产品是：这样能**彻底移除参数沉重的 adaLN-zero**，在深度宽度不变下把模型缩小约 1/3（iMF-Base 从 133M 降到 89M），既省参数又给设计更大模型留了余地，且性能不降反升。

### 损失函数 / 训练策略
最终训练目标是 Flow-Matching 式的 $\mathbb{E}_{t,r,x,e}\lVert V_\theta(z_t)-(e-x)\rVert^2$，其中 $V_\theta$ 采用合法参数化（设计 2）。线性 schedule $z_t=(1-t)x+t\,e$；$V_\theta$ 里的 $u,\frac{d}{dt}u$ 由一次 JVP 同时算出（PyTorch/JAX 的 `jvp` 同时返回函数值与 JVP），$\frac{d}{dt}u$ 上加 stop-gradient。实验设置完全沿用原 MeanFlow 公开代码，在 ImageNet 256×256 类条件生成、预训练 VAE 隐空间（$32\times32\times4$）上从零训练，主评 1-NFE 的 FID-50K。

## 实验关键数据

### 主实验（系统级，ImageNet 256×256，1-NFE，从零训练）

| 配置 | # params | Gflops | FID ↓ | IS ↑ |
|------|------|------|------|------|
| MF-B/2 | 131M | 23.1 | 6.17 | 208.0 |
| MF-XL/2 | 676M | 119.0 | 3.43 | 247.5 |
| iMF-B/2 | 89M | 24.9 | 3.39 | 255.3 |
| iMF-L/2 | 409M | 116.4 | 1.86 | 276.6 |
| **iMF-XL/2** | 610M | 174.6 | **1.72** | **282.0** |

iMF-XL/2 拿到 1.72 FID（相对原 MF 约 50% 提升），且 iMF-B/2 仅 89M 就达到 3.39，已逼平 676M 的 MF-XL/2（3.43）——同等甚至更小规模下大幅领先，全程无蒸馏、无预训练对齐模型。

### 消融实验（MF-B/2 骨架，240 epoch，FID-50K）

| 配置 | FID(w/o CFG) | FID(w/ CFG) | 说明 |
|------|------|------|------|
| 原 MF | 32.69 | 6.17 | 起点 |
| + $V_\theta$，$v_\theta=u_\theta(z_t,t,t)$ 边界条件 | 29.42 | 5.97 | 零额外参数（设计 1+2） |
| + $V_\theta$，辅助 v-head | 30.76 | 5.68 | 推理不增参，w/ CFG 增益更大 |
| + $\omega$-condition（灵活引导） | 25.15 | 5.52 | 单尺度条件（设计 3） |
| + $\Omega$-condition（含 CFG 区间） | 20.95 | 4.57 | 进一步条件化 |
| + in-context 替 adaLN-zero | — | 4.09 | 同时从 133M 降到 89M（设计 4） |
| + 高级 Transformer 块 | — | 3.82 | 架构增益 |
| + 更长训练（640ep） | — | 3.39 | — |

### 关键发现
- **"合法回归目标"是稳训练的关键**：把 JVP 输入从高方差的条件速度 $e-x$ 换成低方差的预测边际速度 $v_\theta$ 后，损失从"高方差、非下降"变成单调下降；w/o CFG 下边界条件变体把 FID 从 32.69 降到 29.42（增益 3.27）。
- **模型越强、合法参数化收益越大**：在更大的 MF-XL/2 上，边界条件变体把 w/ CFG 的 FID 从 3.43 降到 2.99（增益 0.44，相对 B/2 更显著），印证"更有容量的网络能更好地用 $u_\theta(z_t,t,t)$ 学到 $v_\theta$"。
- **灵活引导的价值不全在单点 FID**：$\Omega$-condition 把 FID 一路压到 4.57，但其更大意义是同一模型可在推理时扫不同 $\omega$/区间（图 4），而原 MF 必须训练前定死。
- **in-context 条件注入一举两得**：换掉 adaLN-zero 既把 FID 从 4.57 降到 4.09，又把模型从 133M 砍到 89M。

## 亮点与洞察
- **"换个写法暴露真问题"的范式诊断很漂亮**：作者没有提新损失，而是证明 MF 的 u-loss 等价于被 $u_\theta$ 重参数化的 v-loss，从这个等价视角一眼看穿"目标依赖网络""输入偷吃 $e-x$"两个隐患——重写不改结果却改认知，是方法论上的"啊哈"。
- **零成本 $v_\theta=u_\theta(z_t,t,t)$ 极简却管用**：利用 $r\to t$ 时平均速度退化为瞬时速度这条边界条件，不加任何参数就拿到合法的低方差 JVP 输入。
- **"把超参变成条件"可迁移**：把 CFG 尺度乃至区间当作时间步那样的可学条件注入，这套"超参条件化"思路对任何需要推理时调超参的一步/少步生成模型都通用。
- **in-context 多 token 条件**翻案了 DiT 里"in-context 逊于 adaLN-zero"的结论——只要每条件给多个 token，既补差距又省 1/3 参数。

## 局限与展望
- 验证局限于 ImageNet 256×256 类条件生成这一单一基准，未在文本到图像、更高分辨率或其他模态上检验泛化。
- ⚠️ 辅助 v-head、CFG 条件训练的具体分布与实现细节多在附录，正文给的是结论性数字；边界条件 vs 辅助 v-head 的取舍随模型规模变化，需按规模调。
- stop-gradient 在新框架里"理论可去、实践仍需"，说明优化层面仍有未完全理清之处；更长训练/高级 Transformer 块带来的增益与方法本身的贡献存在一定耦合。

## 相关工作与启发
- **vs 原 MeanFlow**：iMF 不换骨架，只把目标从"网络依赖的 u-loss"改成"网络无关的 v-loss + 合法参数化"，并把定死的 CFG 尺度条件化——直击 MF 的两个根本病根，FID 相对提升约 50%。
- **vs Consistency Models / Shortcut / IMM 等 fastforward 方法**：它们各自用"跳到终点""中点关系""矩匹配"等不同近似构造 look-ahead 目标；iMF 聚焦 MeanFlow 这一支的目标合法性与 CFG 实用性，作者强调这些改进与其它并发改进（AlphaFlow、Decoupled MeanFlow、CMT）正交、可叠加。
- **vs 多步扩散/流匹配**：iMF 用单次 NFE 把 FID 压到 1.72，显著缩小与多步方法的差距，且不依赖蒸馏或预训练对齐模型，支持"fastforward 作为独立范式"的论点。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用等价重写揭示并修复 MF 的两个根本缺陷，视角与解法都干净有力
- 实验充分度: ⭐⭐⭐⭐ 消融逐项拆解、系统级跨规模对比清晰，但限于 ImageNet 单一基准
- 写作质量: ⭐⭐⭐⭐⭐ 推导严谨、图表把"为什么"讲透，损失方差分析尤其到位
- 价值: ⭐⭐⭐⭐⭐ 1-NFE 1.72 FID 逼近多步且不蒸馏，为一步生成确立了更稳的目标范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CaTok: Taming Mean Flows for One-Dimensional Causal Image Tokenization](catok_taming_mean_flows_for_one-dimensional_causal_image_tokenization.md)
- [\[CVPR 2026\] Functional Mean Flow in Hilbert Space](functional_mean_flow_in_hilbert_space.md)
- [\[CVPR 2026\] Stable Mean Flow: Lyapunov-Inspired One-Step Flow Matching](stable_mean_flow_lyapunov-inspired_one-step_flow_matching.md)
- [\[CVPR 2026\] Learning Straight Flows: Variational Flow Matching for Efficient Generation](learning_straight_flows_variational_flow_matching_for_efficient_generation.md)
- [\[CVPR 2026\] Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation](temporal_equilibrium_meanflow_bridging_the_scale_gap_for_one-step_generation.md)

</div>

<!-- RELATED:END -->
