---
title: >-
  [论文解读] Reward Sharpness-Aware Fine-Tuning for Diffusion Models
description: >-
  [CVPR 2026][图像生成][奖励黑客] 本文把扩散模型奖励微调（RDRL）中的"奖励黑客"（reward hacking，奖励分涨但画质不升）诊断为一种"对抗攻击"——奖励模型在其损失面陡峭的方向上不鲁棒；据此提出 RSA-FT，不重训奖励模型，而是改用一个"被抹平"的奖励模型的梯度，做法是在**图像空间**（对抗式输入扰动）和**参数空间**（SAM 式权重扰动）同时施加扰动取局部最差奖励，二者联合即可显著缓解奖励黑客，且能即插即用地嵌进 ReFL / DRaFT-K / AlignProp / DRTune 等各种 RDRL 框架与多种扩散骨干。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "奖励黑客"
  - "扩散模型微调"
  - "Sharpness-Aware"
  - "对抗鲁棒性"
  - "即插即用"
---

# Reward Sharpness-Aware Fine-Tuning for Diffusion Models

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Reward_Sharpness-Aware_Fine-Tuning_for_Diffusion_Models_CVPR_2026_paper.html)  
**代码**: 无  
**领域**: 扩散模型 / 图像生成 / 对齐RLHF  
**关键词**: 奖励黑客, 扩散模型微调, Sharpness-Aware, 对抗鲁棒性, 即插即用

## 一句话总结
本文把扩散模型奖励微调（RDRL）中的"奖励黑客"（reward hacking，奖励分涨但画质不升）诊断为一种"对抗攻击"——奖励模型在其损失面陡峭的方向上不鲁棒；据此提出 RSA-FT，不重训奖励模型，而是改用一个"被抹平"的奖励模型的梯度，做法是在**图像空间**（对抗式输入扰动）和**参数空间**（SAM 式权重扰动）同时施加扰动取局部最差奖励，二者联合即可显著缓解奖励黑客，且能即插即用地嵌进 ReFL / DRaFT-K / AlignProp / DRTune 等各种 RDRL 框架与多种扩散骨干。

## 研究背景与动机
**领域现状**：受 RLHF 在大模型对齐上成功的启发，文图扩散模型也开始用强化学习微调来对齐人类偏好——这一族方法称为奖励中心的扩散强化学习（RDRL，如 ReFL、DRaFT-K、AlignProp、DRTune）。由于训练时实时收集人类反馈不现实，做法是用人类标注训练出的**奖励模型**（如 HPSv2）当人类偏好的可微代理，让扩散模型对它做梯度上升。

**现有痛点**：RDRL 普遍受**奖励黑客**之害——奖励分数一路上涨，但感知质量不升反降（文字扭曲、肢体畸形等）。这一现象在扩散 RL 里此前缺乏系统分析。

**核心矛盾**：奖励模型 $r$ 只是真实人类偏好 $r^\star$ 的近似。当生成器沿着 $r$ 的某些"陡峭方向"优化时，就像对抗攻击——极小的图像扰动能让奖励 logit 暴涨却没有真实画质提升，把更新推进到偏离 $r^\star$ 的孤立非偏好区域。作者由此类比："非鲁棒分类器在分类器引导下会损害样本质量，而对抗训练得到的鲁棒分类器能缓解"，问题是为人类偏好对齐重训一个等价鲁棒的奖励模型代价过高（需要更大模型 + 更多标注）。

**本文目标**：在**不重训**奖励模型的前提下，把它"鲁棒化/抹平"，从而抑制奖励黑客。

**切入角度**：借鉴随机平滑（randomized smoothing）——不重训、靠平滑固定模型的预测就能增强鲁棒性。作者据此观察到"奖励模型恰恰在损失面陡峭处不鲁棒"，于是改用一个被抹平的奖励模型的梯度。更妙的是，这种抹平天然诱导出"最差情况的参数扰动"，与 Sharpness-Aware Minimization（SAM）同源；而 SAM（参数空间）与对抗训练 AT（输入空间）本就存在对偶关系。

**核心 idea**：用"抹平后的奖励模型梯度"代替"原始奖励模型梯度"，并在图像空间和参数空间同时抹平（双重鲁棒），把奖励黑客按下去——这是首个把 AT 与 SAM 统一进 RDRL 框架的工作。

## 方法详解

### 整体框架
方法的本质是把 RDRL 原本"最大化奖励 $r$"的目标，换成"最大化一个抹平后的奖励 $\tilde r^d$"。RDRL 标准目标是 $\mathcal{J}(\theta)=\max_\theta\mathbb{E}_{c,x_T}\big[r(x_0(x_T,c;\theta),c)\big]$，其中 $x_0$ 是从噪声 $x_T$ 出发、条件文本 $c$ 下去噪得到的最终样本。本文把它改成 $\mathcal{J}(\theta)=\max_\theta\mathbb{E}_{c,x_T}\big[\tilde r^d(x_0(x_T,c;\theta),c)\big]$，其中抹平奖励定义为局部邻域内的**最小**奖励 $\tilde r^d(x,c):=\min_{d(x,x')<\rho}r(x',c)$，$d(\cdot,\cdot)$ 是图像流形上的距离度量。作者用两种度量（图像空间、参数空间）各做一步近似，再把两者合并成 RSA-FT 的联合目标。整个方法是一个**即插即用的目标函数替换**：不改扩散骨干、不改奖励模型、不引入额外训练，因此能直接套进任意现有 RDRL 框架。

因为它本质是"对奖励项做对抗/SAM 式正则"的损失改写、而非多阶段或多模块流水线，这里不画框架图，用公式与算法说清即可。其单步算法（Algorithm 1）为：每步采样噪声 $x_T$ 与条件 $c$、生成图像 $x_0$ → 算图像空间扰动 $x_0+\delta_{x_0}$ → 算参数空间扰动 $\theta+\epsilon_\theta$ → 用合并目标更新 $\theta$。

### 关键设计

**1. 奖励陡峭度假设与指标 $S_1$：把奖励黑客量化成"损失面有多陡"**

这是整套方法的诊断地基。作者假设：奖励模型 $r$ 在其奖励面**局部平坦**处泛化最好，**陡峭**处则偏离真实偏好 $r^\star$；奖励黑客就是生成器在 $r$ 的陡峭方向上"钻空子"。为量化，定义奖励陡峭度指标 $S_1=\mathbb{E}_{x\sim\mathcal{D}}\big[r(x)-\min_{\|\epsilon\|<\rho}r(x+\epsilon)\big]$，即奖励在局部邻域内的"跌幅"，可用一步更新近似 $S_1\approx\mathbb{E}_x\big[r(x)-r(x-\rho\frac{\nabla_x r(x)}{\|\nabla_x r(x)\|})\big]$——$S_1$ 越大越陡、越小越平。实证上，作者用 DRaFT-K 微调 SD1.5、以 PickScore / ImageReward 当 $r^\star$ 的代理评估器，发现奖励陡峭度与偏好质量呈强**负相关**（Pearson $r_{corr}=-0.802$ 对 PickScore、$-0.669$ 对 ImageReward），坐实了"陡峭=泛化差=奖励黑客"的假设。

**2. 图像空间抹平：把奖励对图像求"最差扰动"，等价于对奖励做对抗训练**

针对"生成器在图像层面钻奖励空子"，图像空间抹平取邻域内最差奖励 $\max_\theta\mathbb{E}\big[\min_{\|\delta\|<\rho}r(x_0+\delta,c)\big]$。这与对奖励模型做对抗扰动同形：用一步近似，沿奖励对图像的梯度反方向扰动 $\delta_{x_0}=-\rho\frac{\nabla_{x_0}r(x_0,c)}{\|\nabla_{x_0}r(x_0,c)\|}$，于是目标变成 $\max_\theta\mathbb{E}\big[r(x_0+\delta_{x_0},c)\big]$。直觉上，它逼着生成器不能只在"某个像素方向上让奖励虚高"，而要在一个 $\rho$-球邻域内都拿到高奖励，从而把"孤立的非偏好尖峰"压平——这正是随机平滑思想在奖励面上的搬运。

**3. 参数空间抹平：SAM 式权重扰动，惩罚奖励对参数的陡峭方向**

只在图像空间抹平还不够，作者把同一"取局部最差"的原则推到**参数空间**：$\max_\theta\mathbb{E}\big[\min_{\|\epsilon\|<\rho_\omega}r(x_0(x_T,c;\theta+\epsilon),c)\big]$。一步近似给出 SAM 式的权重扰动 $\epsilon_\theta=-\rho_\omega\frac{\nabla_\theta r(x_0,c)}{\|\nabla_\theta r(x_0,c)\|}$，对应目标 $\max_\theta\mathbb{E}\big[r(x_0(x_T,c;\theta+\epsilon_\theta),c)\big]$。这里沿用 SAM 的实践：$\epsilon_\theta$ 虽依赖 $\theta$，但在外层优化时对它做 stop-gradient（不走链式法则）。其作用是促使收敛到奖励面上**平坦的参数极小区**——平坦极小泛化更好，能从权重侧抑制奖励黑客。这一步把 AT（输入空间）与 SAM（参数空间）的对偶关系首次落到 RDRL 上。

**4. RSA-FT 联合目标：图像 + 参数双重抹平，互补放大**

单独用图像空间或参数空间抹平都能独立缓解奖励黑客、提升偏好对齐，但作者发现二者**互补**、合用收益最大。最终联合目标把两种扰动叠加：$\max_\theta\mathbb{E}_{c,x_T}\big[r(x_0(x_T,c;\theta+\epsilon_\theta)+\delta_{x_0},c)\big]$，即在"被权重扰动的模型"生成的样本上，再加一层图像扰动后取奖励——同时在图像和参数两个空间强制平滑，得到"双重鲁棒"的奖励优化。两个扰动半径都搜索 $\{10^{-1},10^{-2},10^{-3}\}$、最优均为 $10^{-2}$。整体仍是即插即用：把原 RDRL 的奖励项替换成这个联合目标即可，不动架构、不加奖励函数。

### 损失函数 / 训练策略
训练即用上文联合目标对扩散参数 $\theta$ 做梯度上升（Algorithm 1 单步）。实现细节：在 H100 上用 AdamW（$\beta_1=0.9,\beta_2=0.999$，weight decay $10^{-4}$）；SD1.5/SDXL 采样 50 步、SD3 采样 28 步；学习率 $2\times10^{-5}$、batch 32；扰动半径 $\rho=\rho_\omega=10^{-2}$；迭代数/epoch 沿用各 baseline 原协议（不为提分额外调参，以纯粹验证 RSA-FT 的增益）。

## 实验关键数据

### 主实验
统一用 HPSv2 当训练奖励信号，把 RSA-FT 嵌进 ReFL / DRaFT-K(K=1) / AlignProp / DRTune，评测用 DrawBench 与 HPSv2 测试集，指标含 HPSv2.1 / PickScore / ImageReward。SD1.5（512×512）上各 baseline 加 RSA-FT 后**全部三项指标同时上涨**：

| 方法（SD1.5 / DrawBench） | HPSv2.1↑ | PickScore↑ | ImageReward↑ |
|------|----------|-----------|--------------|
| Vanilla | 24.02 | 21.02 | -0.147 |
| AlignProp | 25.12 | 20.98 | -0.033 |
| AlignProp + Ours | **29.59 (+4.47)** | **21.51 (+0.53)** | **0.268 (+0.30)** |
| ReFL | 31.08 | 21.57 | 0.536 |
| ReFL + Ours | 31.67 (+0.59) | 21.70 (+0.13) | 0.671 (+0.135) |
| DRTune | 30.63 | 21.34 | 0.477 |
| DRTune + Ours | 31.16 (+0.53) | 21.52 (+0.18) | 0.540 |

在 HPD 子集上增益更明显，如 AlignProp 的 HPSv2.1 从 24.93 提到 32.02（+7.09）、ImageReward 从 0.032 提到 0.528。关键对比点是：AlignProp / Draft-LV 原本只涨 HPSv2.1 而辅助奖励下降（典型奖励黑客），加 RSA-FT 后三项**齐涨**，说明拿到的是真实对齐而非指标过拟合。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅图像空间扰动 | 独立改善 | 单独即可缓解奖励黑客、提升偏好对齐 |
| 仅参数空间扰动 | 独立改善 | 同上，SAM 式权重平滑单独亦有效 |
| 图像 + 参数（RSA-FT） | 增益最大 | 二者互补、协同效应明显（详表见原文附录 E / Table 6） |

注：完整消融数值原文放在附录 ⚠️（以原文为准，正文只给定性结论）。

### 关键发现
- **奖励黑客 ≈ 奖励面陡峭**：陡峭度 $S_1$ 与人类偏好强负相关（$-0.802$/$-0.669$），是全文最硬的证据，把一个直觉现象量化成可测指标。
- **跨骨干/分辨率/架构稳健**：从 SD1.5(512²) 到 SDXL/SD3(1024²) 再到 MMDiT 结构的 Flux.1-dev，RSA-FT 都能持续提分，说明"抹平奖励"的策略与具体骨干无关。
- **人类研究佐证**：17 名标注者的偏好研究里，加 RSA-FT 后在视觉/文本偏好上多数超过 50% 阈值（如 SD3+ReFL 视觉偏好 65.4% vs 34.6%），但作者自承样本量小、仅作支撑证据。

## 亮点与洞察
- **一个"抹平"统一了 AT 与 SAM 两条鲁棒性脉络**：图像空间扰动 = 对奖励做对抗训练，参数空间扰动 = SAM，本文指出它们在 RDRL 里同源、且互补——这种"对偶视角"很优雅，也解释了为何合用最强。
- **不重训奖励模型这一点极具工程价值**：构造同等鲁棒的奖励模型需大模型 + 大量标注，本文只改优化目标、零额外训练就拿到鲁棒奖励梯度，真正即插即用。
- **把"奖励黑客"形式化为对抗攻击**很有启发：它把生成式对齐里一个模糊的失败模式，接到了成熟的对抗鲁棒性 / 随机平滑 / SAM 工具箱上，后续可复用大量现成理论与技巧。

## 局限与展望
- 作者承认：评测主要依赖基于模型的指标（HPSv2/PickScore/ImageReward），它们本身只是人类偏好的不完美代理；人类研究规模小（17 人）、统计功效不足，只能当支撑证据。
- 当前只在**单奖励模型**下研究奖励黑客；多奖励设置下抹平能否互补补偿各模型弱点，仅作展望未验证。
- 抹平用的是一步**最小化**近似，作者也指出高斯平均等替代平滑可能更鲁棒，但为效率选了一步法 ⚠️（以原文为准）。
- 目前对所有样本**一视同仁**地施加抹平奖励；未来可探索"选择性 sharpness 加权"，对过陡样本降权——这是作者点出的明确改进方向。

## 相关工作与启发
- **vs 对抗训练 AT**：AT 在输入空间求最差损失提升鲁棒性，但需重训且常需更大模型；本文把"输入空间扰动"搬到奖励上、不重训奖励模型，只改生成器优化目标。
- **vs SAM（Sharpness-Aware Minimization）**：SAM 在参数空间促平坦极小提升泛化；本文把它用到 RDRL 的奖励优化，且与图像空间扰动联合——SAM 此前在 RL 和扩散里被各自单独用过，本文首次在 RDRL 里把两视角统一。
- **vs 随机平滑 / AWP**：随机平滑靠高斯平均不重训地增鲁棒，启发了本文的"抹平奖励"；AWP 在图像分类里同时用输入+权重扰动，本文是首个把这一原则延伸到 RDRL 框架的工作。
- **vs 现有 RDRL（ReFL/DRaFT-K/AlignProp/DRTune）**：它们直接最大化奖励、易奖励黑客；RSA-FT 不替换它们，而是作为即插即用模块叠加，普遍提升其鲁棒性与对齐质量。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把奖励黑客诊断为对抗攻击、并用 AT+SAM 对偶统一抹平奖励，视角新且自洽。
- 实验充分度: ⭐⭐⭐⭐ 跨 4 框架 × 4 骨干 × 2 基准全面验证，但完整消融在附录、人类研究规模小。
- 写作质量: ⭐⭐⭐⭐⭐ 从假设→指标→实证负相关→方法的逻辑链清晰，公式与几何直觉到位。
- 价值: ⭐⭐⭐⭐⭐ 零额外训练、即插即用、跨骨干稳健，对扩散对齐社区实用性很强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Do Less, Achieve More: Do We Need Every-Step Optimization for RL Fine-tuning of Diffusion Models?](do_less_achieve_more_do_we_need_every-step_optimization_for_rl_fine-tuning_of_di.md)
- [\[CVPR 2026\] CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think](craft_aligning_diffusion_models_with_finetuning_is_easier_than_you_think.md)
- [\[CVPR 2026\] Towards Fine-Grained Attribution: Instance-Aware Preference Optimization for Aligning Diffusion Models](towards_fine-grained_attribution_instance-aware_preference_optimization_for_alig.md)
- [\[CVPR 2025\] Personalized Preference Fine-tuning of Diffusion Models](../../CVPR2025/image_generation/personalized_preference_fine-tuning_of_diffusion_models.md)
- [\[CVPR 2026\] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)

</div>

<!-- RELATED:END -->
