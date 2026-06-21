---
title: >-
  [论文解读] NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching
description: >-
  [ICLR 2026][多模态VLM][离散流匹配] 用离散流匹配（Discrete Flow Matching, DFM）取代自回归（AR）作为统一建模范式，搭出第一个完全基于 DFM 的开源全模态基础模型 NExT-OMNI，单编码器统一表示同时支撑文/图/视频/音的理解、生成与跨模态检索。 - 领域现状：统一的多模态理…
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "离散流匹配"
  - "全模态"
  - "any-to-any"
  - "统一表示"
  - "跨模态检索"
---

# NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=odatOcBi61](https://openreview.net/forum?id=odatOcBi61)  
**代码**: [https://github.com/ritzz-ai/Next-OMNI](https://github.com/ritzz-ai/Next-OMNI)  
**领域**: 多模态 / 全模态基础模型  
**关键词**: 离散流匹配, 全模态, any-to-any, 统一表示, 跨模态检索  

## 一句话总结
用离散流匹配（Discrete Flow Matching, DFM）取代自回归（AR）作为统一建模范式，搭出第一个完全基于 DFM 的开源全模态基础模型 NExT-OMNI，单编码器统一表示同时支撑文/图/视频/音的理解、生成与跨模态检索。

## 研究背景与动机
- **领域现状**：统一的多模态理解+生成被视为通往 AGI 的关键瓶颈。主流做法是把大语言模型的自回归成功经验搬到多模态上，再用混合架构或模块解耦（Janus、Show-o、Bagel 等）分别处理理解和生成。
- **现有痛点**：AR 范式在理解与生成任务间存在内在冲突，难以平衡两者；混合/解耦方案本质是"按任务路由"，引入冗余参数化模块、结构复杂、推理慢，而且编码特征过度分离，**无法产出真正统一的表示**，导致跨模态检索这类需要深度特征融合的任务表现糟糕。
- **核心矛盾**：解耦能换来单项任务表现，却以牺牲统一性和通用性为代价；想要 any-to-any、跨模态检索这种"广义多模态"能力，又恰恰需要融合而非解耦。
- **本文目标**：找一条"融合"而非"解耦"的统一建模路径，让一个精简架构同时把理解、生成、检索都吃下，并且推理更快。
- **核心 idea**：**用离散流匹配做统一范式**——DFM 从全损坏序列出发并行迭代去噪，天然具备双向信息整合和并行解码加速；配合**重建增强的统一表示**，让单个编码器既保留高层语义又保留低层细节，从而支撑深度特征融合下的检索与多轮交互。

## 方法详解

### 整体框架
NExT-OMNI 把视觉、文本、音频统一编码成离散 token 序列，用一个从 AR-LLM（Qwen2.5-7B）初始化的骨干在每层做多模态自注意力深度融合，整个序列在离散流匹配范式下并行去噪预测，再由轻量的模态专属 head 解码回各模态。与"多编码器 + MoE/MoT 解耦"的前作不同，它训练**单一编码器同时服务理解与生成**，产出可直接用于检索的统一表示。

```mermaid
flowchart LR
    A[文本/图像/视频/音频输入] --> B[模态编码器<br/>CLIP-ViT / Whisper + VQVAE]
    B --> C[离散 token 序列 x1]
    C --> D[加噪 xt ~ pt·|x1]
    D --> E[共享骨干<br/>每层多模态自注意力<br/>DFM 速度预测]
    E --> F[模态专属 Head<br/>LM/Vision/Audio]
    F --> G[Any-to-Any 生成]
    E --> H[EOS token 特征<br/>跨模态检索]
```

### 关键设计

**1. 离散流匹配统一建模：把 any-to-any 变成一次并行去噪。** 给定从目标分布采样的"视觉-文本-音频"交错序列，先用 VQVAE 编码器和文本 tokenizer 转成离散目标 token 序列 $x_1=(x_1^1,\dots,x_1^D)$，每步均匀采样时刻 $t\in[0,1]$、按概率路径 $p_t(\cdot|x_1)$ 采出带噪序列 $x_t$，模型吃进 $x_t$ 并预测 $x_1$，损失就是逐 token 的期望交叉熵 $L_{ce}=\mathbb{E}_{t,x_1,x_t}\big[-\sum_{i=1}^{D}\log p_{1|t}(x_1^i|x_t)\big]$。这一范式让所有模态共享同一套去噪目标，省掉了 AR 的因果掩码与生成专用的 diffusion/flow head，只留轻量 token 解码头，既加速训练又加速响应。一个实现巧思是：模型并不直接喂离散 token，而是**从编码器码本里抽出带丰富语义和细节的连续表示向量 $c^M_{z_q}$**，经轻量投影与文本嵌入对齐后再送入骨干，简单却显著提升后续优化效果。

**2. 重建增强的统一表示：单编码器既要语义又要细节。** 先做 warmup 训练，让视觉/音频编码器在两个目标下学统一表示——一是借助辅助 VQVAE 量化器和模态解码器的重建损失 $L^M_{rec}=L^M_R+L^M_{VQ}+L^M_G$（量化为 $z_q=\arg\min_{c\in C^M}\|z^M-c\|_2$，捕捉低层细节），二是语义对齐损失 $L^M_{sem}$（视觉用 CLIP 式句级对比 $L^V_{constra}$、音频用 Whisper 式 token 级 caption 对齐 $L^A_{cap}$），合成 $L^M_{total}=L^M_{rec}+L^M_{sem}$。关键在于 DFM 训练时**复用这套重建损失**，把总目标写成 $L_{overall}=\lambda_1 L_{ce}+\lambda_2 L^V_{rec}+\lambda_3 L^A_{rec}$，并用 GradNorm 动态平衡三项梯度贡献。这样防止模型在 DFM 训练中过度偏向高层语义而丢掉细粒度信息，既保住理解/生成质量，又让统一表示足够"深融合"，从而支撑基于特征相似度的精准跨模态检索——检索时直接取 \<EOS\> token 特征即可。

**3. 动态长度生成策略（DGS）：补上 DFM 在理解任务上的短板。** DFM 并行去噪天生不擅长变长文本生成，作者在训练时插入额外 \<PAD\> 让响应序列对齐到 block size 的整数倍；推理时利用"简单 token 单步即可确定"的性质，根据 \<EOS\> 置信度以 block size 为步长动态扩展到合适的预设长度，再做多步迭代去噪。这一招以极小代价把理解任务（如 VQA）拉回到与 AR 相当的水平，消融里 DGS 让 VQAv2 从 51.7 直接回升到 54.3。

**4. 自适应缓存 + 交错单模态训练：把"更快"落到实处。** 推理时观察到多步去噪中大部分特征变化极小（类 DLLM 现象），于是对指令部分整段缓存并最小更新，对响应部分按"value 特征与缓存特征的余弦相似度"自适应更新，配合 DFM 的并行解码，相比 AR 架构拿到 1.2× 响应加速。训练侧则规避"随机混合多模态导致负载不均"的浪费——**每个 batch 只训一个模态**，靠多任务交错 + 梯度累积达成联合目标，换来 1.4× 训练效率提升。

## 实验关键数据

### 主实验表格

全模态理解（三个 benchmark 平均，AR 对手中最强 OpenOmni 36.5）：

| 模型 | OmniBench(T+A+V) | WorldSense(T+A+V) | AV-Odyssey | AVG. |
|------|------|------|------|------|
| OpenOmni | 37.4 | 37.2 | 32.8 | 36.5 |
| **NExT-OMNI** | **40.7** | **40.5** | **36.4** | **39.7** |

相比 OpenOmni 三数据集平均 +3.2 绝对提升。多轮语音交互（Spoken QA，S→T 平均）NExT-OMNI 62.0 居首；多轮视觉交互（OpenING）AVG. 55.0 显著超过同架构 MMaDA(47.7)、FUDOKI(44.5) 与 AR 系 SEED-X(50.2)。

### 消融实验表格

| Paradigm | Rep. | DGS | Recon. | VQAv2(Und) | GenEval(Gen) | InfoSeek(Ret) | AVG. |
|------|------|------|------|------|------|------|------|
| AR | Decoupled | × | × | 55.2 | 53.4 | 28.3 | 41.4 |
| DFM | Decoupled | × | × | 52.3 | 59.8 | 29.6 | 42.6 |
| DFM | Unified | × | × | 51.7 | 59.2 | 32.8 | 43.0 |
| DFM | Unified | ✓ | × | 54.3 | 59.4 | 33.1 | 43.9 |
| DFM | Unified | ✓ | ✓ | **56.2** | **62.6** | **33.7** | **45.6** |

### 关键发现
- **范式替换有得有失**：AR→DFM 让理解微降但生成、检索明显上升；再转统一表示时理解因粒度冲突进一步下滑，但检索持续受益——印证"统一表示更适合广义应用"。
- **DGS 救理解**：动态长度生成把理解拉回与 AR 可比，VQAv2 +2.6。
- **重建项救生成与检索**：复用重建损失给特征补低层细节、抑制对高层语义的过度偏置，生成 GenEval +3.2、整体 AVG. 升到 45.6。
- **检索上 DFM/扩散类 > AR/混合类**：双向信息编码退化成 BERT 式特征提取，比因果掩码能更好聚合上下文，解耦方案因特征过度分离反而检索吃亏。

## 亮点与洞察
- **范式层面的"第一个"**：首个完全基于离散流匹配的开源全模态基础模型，把 DFM 从单模态语言/图像建模推广到 text-image-video-audio 的统一 any-to-any。
- **统一表示的额外红利**：把"理解+生成"的统一表示直接用于跨模态检索，揭示了一个被解耦路线牺牲掉的应用维度——检索性能反过来成为衡量"表示是否真统一"的探针。
- **工程与范式互补**：重建损失复用、GradNorm 平衡、单模态交错训练、自适应缓存，都是围绕"DFM 弱在哪就补哪"的针对性设计，把范式优势真正兑现为速度与多任务覆盖。

## 局限与展望
- **理解仍需 DGS 补救**：DFM 在变长文本生成上的固有短板靠工程策略弥补，说明纯 DFM 在语言密集型理解上尚未天然胜过 AR。
- **视频生成偏初步**：长视频靠"抽 8 帧当多图"的简化策略，video/long-audio 生成仍是"preliminary"级别。
- **检索训练规模有限**：仅在 M-BEIR 100K 子集上做检索微调，更大规模与更多检索场景的泛化待验证。
- **作者展望**：把 NExT-OMNI 推向 VLA 的动作轨迹生成与世界模型的物理视频生成等更广领域。

## 相关工作与启发
- **对照 AR/混合统一模型**：Janus（AR 解耦）、Show-o（AR+离散扩散）、Bagel（AR+Diffusion 解耦 MoT），本文用"DFM + 统一表示"在检索维度上系统性胜出，提供了"融合优于解耦"的实证。
- **对照同范式工作**：MMaDA（离散扩散）、FUDOKI（DFM 但解耦），NExT-OMNI 证明 DFM 配统一表示比 DFM 配解耦更通用。
- **启发**：当评估只看理解+生成时容易低估解耦方案的代价；引入跨模态检索作为评测维度，能更公平地暴露"表示是否真正统一"，这对未来统一多模态模型的 benchmark 设计有参考价值。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 首个全 DFM 全模态基础模型，把离散流匹配从单模态推广到 any-to-any 并用统一表示打通检索，范式选择有清晰动机。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖理解/语音交互/视觉交互/检索四类任务 + 关键组件消融，对照面广；但视频生成、检索规模偏弱。
- **写作质量**: ⭐⭐⭐⭐ 动机与"解耦 vs 融合"主线清晰，图 2 pipeline 和消融逐项归因到位。
- **价值**: ⭐⭐⭐⭐ 为统一多模态建模提供了一条可复现的非 AR 路线（开源代码+权重），并把跨模态检索抬为衡量统一性的有用探针。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] FlowBind: Efficient Any-to-Any Generation with Bidirectional Flows](flowbind_efficient_any-to-any_generation_with_bidirectional_flows.md)
- [\[NeurIPS 2025\] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification](../../NeurIPS2025/multimodal_vlm/mdreid_modality-decoupled_learning_for_any-to-any_multi-modal_object_re-identifi.md)
- [\[CVPR 2026\] Describe Anything Anywhere At Any Moment](../../CVPR2026/multimodal_vlm/describe_anything_anywhere_at_any_moment.md)
- [\[ICLR 2026\] Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs](grasp_any_region_towards_precise_contextual_pixel_understanding_for_multimodal_l.md)
- [\[ICLR 2026\] Omni-Captioner: Data Pipeline, Models, and Benchmark for Omni Detailed Perception](omni-captioner_data_pipeline_models_and_benchmark_for_omni_detailed_perception.md)

</div>

<!-- RELATED:END -->
