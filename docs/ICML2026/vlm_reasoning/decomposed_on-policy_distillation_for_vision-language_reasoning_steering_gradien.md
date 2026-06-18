---
title: >-
  [论文解读] Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding
description: >-
  [ICML2026 Spotlight][多模态VLM][在线蒸馏] 作者把多模态在线蒸馏的 KL 损失沿贝叶斯链拆成"语言先验"和"视觉接地"两个子目标，发现两者梯度近乎正交、标准蒸馏只是被动取平分，提出 Visual Gradient Steering（VGS）主动把更新方向偏向视觉子空间，在 Qwen3-VL 8B→2B/4B 七个多模态推理基准上平均提升 +2.37%/+1.56%。
tags:
  - "ICML2026 Spotlight"
  - "多模态VLM"
  - "在线蒸馏"
  - "视觉接地"
  - "梯度正交"
  - "视觉梯度引导"
  - "VLM 推理"
---

# Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding

**会议**: ICML2026 Spotlight  
**arXiv**: [2606.00564](https://arxiv.org/abs/2606.00564)  
**代码**: https://github.com/hee-suk-yoon/Decomposed_OPD  
**领域**: 多模态VLM / LLM推理  
**关键词**: 在线蒸馏, 视觉接地, 梯度正交, 视觉梯度引导, VLM 推理  

## 一句话总结
作者把多模态在线蒸馏的 KL 损失沿贝叶斯链拆成"语言先验"和"视觉接地"两个子目标，发现两者梯度近乎正交、标准蒸馏只是被动取平分，提出 Visual Gradient Steering（VGS）主动把更新方向偏向视觉子空间，在 Qwen3-VL 8B→2B/4B 七个多模态推理基准上平均提升 +2.37%/+1.56%。

## 研究背景与动机

**领域现状**：小模型获取推理能力的两条主流路径是 RLVR 和在线蒸馏（On-Policy Distillation）。在线蒸馏由教师模型对学生自己采样的轨迹做 token 级稠密监督，规避了 RLVR 稀疏奖励的冷启动问题，已经在纯文本 LLM 上验证有效。

**现有痛点**：把在线蒸馏直接搬到 VLM 时，业界要么完全回避（如 Qwen3-VL 显式限制只蒸馏纯文本数据来微调 LLM backbone，放弃视觉对齐），要么粗暴地把多模态条件 KL 当成一个单一目标拟合，导致视觉接地能力得不到充分迁移。

**核心矛盾**：作者通过 Bayes 分解 $\log p(\tau\mid I,x)=\log p(\tau\mid x)+\log p(I\mid\tau,x)-\log p(I\mid x)$ 揭示，整体 KL 实际包含"语言先验对齐"与"视觉接地对齐"两个子目标；通过梯度几何分析发现两者在高视觉依赖 token 上几乎正交（视觉依赖最高的第 9 桶里夹角 $\approx 92^\circ$），标准单体损失梯度始终保持在两者之间约 $42^\circ$~$50^\circ$ 的折中方向上，相当于一个静态平分线，对真正卡住的视觉子空间投入不足。

**本文目标**：在不打破语言先验的前提下，主动把优化方向向视觉子空间倾斜，把更新预算花在解决感知歧义而非通用语言建模上。

**切入角度**：作者提出"非对称成熟假设"——经过预训练 + GRPO 的 VLM 学生其实已经具备较强的语言先验，真正的瓶颈是视觉感知。如果两个梯度本来就正交，那"加权求和"在视觉极端区会被语言项无意义地稀释；显式把更新方向往视觉旋转就是免费的提升。

**核心 idea**：在标准 KL 之上加一个针对"视觉信息增益"的额外 KL 项 $\gamma\ell_{\text{Vis}}$，并对高视觉依赖 token 额外加一个语言保留正则 $\lambda\ell_{\text{LP}}$，再用梯度范数归一化常数 $\eta_{\text{VGS}}(\gamma)$ 保证只改方向不改步长。

## 方法详解

### 整体框架

VGS 沿用在线蒸馏的标准设置：学生策略 $p_S^\theta$ 在多模态输入 $(I,x)$ 下自采样轨迹 $\tau$，教师 $q_T$（GRPO 微调过的 8B Qwen3-VL）给出 token 级监督。pipeline 与标准做法的差异只发生在 loss 上：原本是一个 multimodal Reverse KL $\ell_{\text{Standard}}(\tau)$，现在变成三项加和 $\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)+\lambda\ell_{\text{LP}}(\tau)$，并乘以一个梯度范数归一化系数 $\eta_{\text{VGS}}(\gamma)$。轨迹采样、模型架构、训练框架、batch 安排都没动，工程复用成本极低。

### 关键设计

**1. 视觉信息增益子目标 $\ell_{\text{Vis}}$：给"视觉接地"一个能单独优化的抓手**

标准蒸馏把多模态 KL 当成一坨，视觉项被语言项稀释。作者先把它拆开：由 Bayes 链式法则 $\log p(I\mid\tau,x)=[\log p(\tau\mid I,x)-\log p(\tau\mid x)]+\log p(I\mid x)$，对教师做同样分解，就能构造出一个"目标分布"$q_T^*$——它保留学生自己的语言先验，只把视觉似然换成教师的：$q_T^*(\tau\mid I,x)\propto p_S^\theta(\tau\mid x)\cdot q_T(I\mid\tau,x)$。在 logit 空间这等价于

$$\log q_T^*=\log p_S^\theta(\tau\mid x)+\big(\log q_T(\tau\mid I,x)-\log q_T(\tau\mid x)\big)-\log Z^*,$$

每一项都只要让学生 / 教师在"多模态上下文"和"纯文本上下文"各跑一次前向就能拿到，不需要额外网络。于是视觉子目标就是学生对这个目标分布的 KL：$\ell_{\text{Vis}}(\tau)=\frac{1}{|\tau|}\sum_t D_{KL}(p_S^\theta(\cdot\mid\tau_{<t},I,x)\,\|\,q_T^*(\cdot\mid\tau_{<t},I,x))$，只惩罚"视觉信息增益"上的差距。之所以非要把它单独拎出来，是因为实证发现视觉依赖越高的 token，$\nabla\ell_{\text{Lang}}$ 与 $\nabla\ell_{\text{Vis}}$ 的夹角从 $\sim60^\circ$ 单调拉到 $\sim92^\circ$——两个梯度几何上几乎独立，靠单体 KL 被动平均就永远喂不饱视觉那一支。

**2. 视觉梯度引导 + 范数归一化 $\eta_{\text{VGS}}(\gamma)$：只转方向，不动步长**

有了独立的视觉项，就可以把更新方向显式往视觉子空间旋：令 $\ell_{\text{VGS}}(\tau)=\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)$，$\gamma\ge 0$ 控制旋转幅度。但直接加权有个副作用——$\gamma$ 在改方向的同时也把梯度模长放大了，相当于偷偷改了学习率，调参时方向和步长会互相污染。作者用一个归一化常数把步长按回去：

$$\eta_{\text{VGS}}(\gamma)=\frac{\|\nabla_\theta\mathcal{L}_{\text{Standard}}\|_2}{\|\nabla_\theta\mathcal{L}_{\text{Standard}}+\gamma\nabla_\theta\mathcal{L}_{\text{Vis}}\|_2},$$

乘到整体 loss 上后，steered 梯度的范数始终等于标准梯度的范数，于是 $\gamma$ 就成了纯粹的"方向旋钮"。实操中这个比值在训练全程几乎不变，作者干脆固定成常数（2B 用 $\eta=0.41$，4B 用 $\eta=0.36$）省掉动态计算。把方向与步长解耦本是 multi-task learning（如 GradNorm）的稳健性常识，这里它让 $\gamma$ 消融变得干净可控。

**3. 语言保留正则 $\ell_{\text{LP}}$：在会"反咬"的少数 token 上守住语言先验**

视觉 steering 不是处处安全。几何分析显示在视觉依赖最极端的那一桶里，$\nabla\ell_{\text{Lang}}$ 与 $\nabla\ell_{\text{Vis}}$ 已经成钝角（$>90^\circ$），此时往视觉方向使劲会反向把语言先验匹配度拉低，训练曲线上能直接看到 $\ell_{\text{Lang}}$ 发散。作者的对策是只在这批 token 上加一道语言 KL 正则——按视觉依赖度 VDS（$\text{VDS}_t=D_{KL}(q_T(\cdot\mid\tau_{<t},I,x)\,\|\,q_T(\cdot\mid\tau_{<t},x))$）取 top 30%，对它们施加 $\ell_{\text{LP}}(\tau)=\frac{1}{|\tau|}\sum_t \mathbf{1}[\text{VDS}_t>Q_{0.7}]\cdot D_{KL}(p_S^\theta(\cdot\mid\tau_{<t},x)\,\|\,q_T(\cdot\mid\tau_{<t},x))$，权重 $\lambda\approx 0.01$ 很保守。只保护这少数真正会反咬的 token，既挡住了灾难性遗忘，又不会对一般 token 加重监督、稀释掉视觉 steering 的力度。

### 损失函数 / 训练策略

最终训练目标为 $\mathcal{L}_{\text{VGS-LP}}=\eta_{\text{VGS}}(\gamma)\cdot\mathbb{E}_{\tau\sim p_S^\theta(\cdot\mid I,x)}[\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)+\lambda\ell_{\text{LP}}(\tau)]$。所有实验固定 $\gamma=2.0$、$\lambda=0.01$。教师由 Qwen3-VL-8B-Instruct 在 Vision-SR1-47K 上用 GRPO 训练 2 epoch 得到；学生为 Qwen3-VL-2B/4B-Instruct，强制使用统一 system prompt 把推理链与最终答案分隔开。同样的 loss 也可以无缝挂到 GRPO 训练循环上做 RL+蒸馏混合。

## 实验关键数据

### 主实验

主结果：Qwen3-VL-8B 教师在 Vision-SR1-47K 上蒸馏到 2B / 4B 学生，7 个多模态推理基准的平均 Acc@1（greedy）和 Acc@16（T=1.0）。

| 设置 | Teacher 8B | Standard Distill Acc@1 | VGS Acc@1 | 提升 |
|------|-----------|-----------------------|-----------|------|
| 2B 学生平均（7 benchmark） | 61.37 | 43.74 | 46.10 | +2.37 |
| 4B 学生平均（7 benchmark） | 61.37 | 56.64 | 58.12 | +1.56 |
| 2B / VisualPuzzles | 43.15 | 28.08 | 31.76 | +3.68 |
| 2B / LogicVista | 60.01 | 45.53 | 48.88 | +3.35 |
| 2B / MathVerse-VD | 79.63 | 56.02 | 58.10 | +2.08 |
| 4B / MathVerse-VD | 79.63 | 71.53 | 74.31 | +2.78 |
| 4B / MathVision | 44.14 | 37.96 | 40.59 | +2.63 |

教师-学生容量差越大，VGS 增益越显著（2B 平均 +2.37 vs 4B +1.56），与"视觉感知是真正瓶颈"的假设一致。

### 消融实验

GRPO + 蒸馏 ablation（2B 学生，Vision-SR1-47K）：

| 配置 | 平均 Acc@1 | 平均 Acc@16 | 说明 |
|------|-----------|-------------|------|
| Initial Student (2B) | 31.32 | – | 起点 |
| 纯 GRPO | 44.83 | 45.68 | 单 RL baseline |
| GRPO + Standard-KD | 45.41 | 45.22 | 加单体蒸馏 |
| GRPO + VGS (full) | 47.20 | 46.57 | +1.79 / +1.35 over Standard-KD |

### 关键发现

- 视觉依赖越高的 token，VGS 对 $\ell_{\text{Vis}}$ 的下降速度越快（图 4），证实方向旋转的确把更新预算花在了视觉子空间。
- 关闭 LP 正则、把 $\gamma$ 拉到 2.0，高 VDS 桶的 $\ell_{\text{Lang}}$ 会显著上涨；加入 LP 后曲线被压回，多模态推理精度不掉，证明 LP 的选择性激活是必要的。
- 反向"Language Steering"实验显示，把 steering 方向旋转到语言子空间反而拉低平均精度，直接证伪了"对称投资"的朴素想法，验证了"视觉子空间才是瓶颈"的非对称假设。
- $\eta_{\text{VGS}}$ 范数归一化非常关键：去掉它后改变 $\gamma$ 等价于同时改学习率，超参很难调。

## 亮点与洞察

- **把模型蒸馏问题拉回到优化几何视角**：以往讨论 KL 蒸馏多停留在概率层面（前向 vs 反向 KL、mode-covering vs mode-seeking），本文转而看梯度方向的余弦关系，揭示了"标准 KL = 静态平分线"这种以前没人讲清的结构性缺陷，思想可迁移到 audio-language、video-language 等任何多条件生成的蒸馏场景。
- **可计算的视觉信息增益**：构造目标分布 $q_T^*$ 只用到学生 / 教师在多模态 + 纯文本两个上下文上的 logit 比值，不需要任何额外网络或额外采样，对工程友好；只要 base model 支持"图像 token 可摘除"就能直接用。
- **VDS 分桶分析**：用 $\text{VDS}_t=D_{KL}(q_T(\cdot\mid I,x)\,\|\,q_T(\cdot\mid x))$ 衡量 token 级视觉依赖度，是一个非常干净的度量，可作为 VLM 可解释性工具单独使用（如做训练数据筛选、注意力分析）。

## 局限与展望

- 实验仅覆盖 Qwen3-VL 系列（2B/4B/8B），没在 LLaVA、InternVL、Gemma 等架构上交叉验证 $q_T^*$ 构造的稳定性；不同 base model 的 vision-text 融合方式差异较大，结论的可迁移性需进一步验证。
- "纯文本上下文"在 VLM 上的意义其实需要假设模型在没有 image token 时仍能给出合法分布，对那些用学习型 image token / soft prompt 的架构（如 Q-Former 加在词嵌入之前）未必成立。
- $\gamma=2.0$、$\lambda=0.01$、阈值 $Q_{0.7}$ 都是经验值，没给出按学生大小自动调参的方案；尤其在 RL 阶段加入蒸馏时 $\alpha$ 的设置可能与 $\gamma$ 强耦合。
- 当前评测均为静态图像 + math/logic 推理任务，VGS 对真实多模态对话、长视频理解等场景的迁移性未知。

## 相关工作与启发

- **vs On-Policy Distillation (Agarwal et al., 2024)**：他们提出 Reverse KL + on-policy 采样的纯文本蒸馏框架，本文是其多模态延伸；区别是直接攻击单体 KL 的几何缺陷，而不是改 KL 类型。
- **vs GradNorm / PCGrad / GradVac 等 multi-task gradient surgery**：传统做法是 per-task 梯度后做投影 / 重加权，本文不需要 per-task 梯度（视觉与语言项共享同一批 token），开销几乎为零，且方向修正是显式可解释的（始终向 $\nabla\ell_{\text{Vis}}$ 倾斜）。
- **vs RLVR / GRPO**：RL 提供稀疏的 outcome 奖励，蒸馏提供稠密的 token 级监督，两者本文是互补关系；GRPO + VGS 在 length 上得到了与教师对齐的隐式正则化（图 6），相比纯 GRPO 的 length explosion 更稳定。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Bayes 分解 + 梯度正交分析是 VLM 蒸馏文献里第一次把"语言/视觉"显式拆开看梯度几何的工作。
- 实验充分度: ⭐⭐⭐⭐ 七个推理 benchmark + 两个学生规模 + GRPO 混合训练，但只覆盖一个模型家族。
- 写作质量: ⭐⭐⭐⭐⭐ 推导清晰、几何图 (Fig.3) 与训练动力学图 (Fig.4) 自洽地服务于核心假设。
- 价值: ⭐⭐⭐⭐ 工程上几乎零成本就能挂到现有 on-policy 蒸馏框架，对 small VLM 训练社区是即插即用的提升。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation](../../CVPR2026/multimodal_vlm/vold_reasoning_transfer_from_llms_to_vision-language_models_via_on-policy_distil.md)
- [\[CVPR 2026\] PDCR: Perception-Decomposed Confidence Reward for Vision-Language Reasoning](../../CVPR2026/multimodal_vlm/pdcr_perception-decomposed_confidence_reward_for_vision-language_reasoning.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[CVPR 2026\] CodeV: Code with Images for Faithful Visual Reasoning via Tool-Aware Policy Optimization](../../CVPR2026/multimodal_vlm/codev_code_with_images_for_faithful_visual_reasoning_via_tool-aware_policy_optim.md)
- [\[CVPR 2026\] Multimodal Distribution Matching for Vision-Language Dataset Distillation](../../CVPR2026/multimodal_vlm/multimodal_distribution_matching_for_vision-language_dataset_distillation.md)

</div>

<!-- RELATED:END -->
