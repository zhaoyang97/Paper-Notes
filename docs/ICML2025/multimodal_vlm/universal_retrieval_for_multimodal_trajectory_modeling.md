---
title: >-
  [论文解读] Universal Retrieval for Multimodal Trajectory Modeling
description: >-
  [ICML 2025][多模态VLM][trajectory retrieval] 首次系统定义多模态轨迹检索任务，构建统一代理轨迹数据集 UATD（7,747 个演示、82,793 个状态）和 GAE-Bench 基准（714,628 正样本对），提出基于 VLM2Vec 的 GAE-Retriever 框架，在 5 个 GUI 环境上相比最强基线 VLM2Vec-V2.2 平均提升 10.22 个百分点。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "trajectory retrieval"
  - "GUI agents"
  - "多模态"
  - "对比学习"
  - "VLM"
---

# Universal Retrieval for Multimodal Trajectory Modeling

**会议**: ICML 2025  
**arXiv**: [2506.22056](https://arxiv.org/abs/2506.22056)  
**代码**: 无  
**领域**: 多模态 / GUI Agent  
**关键词**: trajectory retrieval, GUI agents, multimodal embedding, contrastive learning, VLM

## 一句话总结

首次系统定义多模态轨迹检索任务，构建统一代理轨迹数据集 UATD（7,747 个演示、82,793 个状态）和 GAE-Bench 基准（714,628 正样本对），提出基于 VLM2Vec 的 GAE-Retriever 框架，在 5 个 GUI 环境上相比最强基线 VLM2Vec-V2.2 平均提升 10.22 个百分点。

## 研究背景与动机

**轨迹数据的价值与挑战**：人机交互中记录的轨迹数据（指令视频、操作指南、GUI 导航记录）蕴含丰富的状态-动作序列知识，对 in-context reasoning、强化学习、世界建模等下游任务有巨大价值。随着 AI agent 产品部署和研究推进，轨迹数据量正在爆发式增长。

**现有方法的不足**：当前利用轨迹数据的方法（如从 memory 中检索可复用子程序）仅使用文本特征做相似性搜索，**忽略了多模态信号**（截图、UI 布局等）。更重要的是，现有工作缺乏对轨迹检索的系统性任务定义、统一数据格式和标准化基准。

**核心问题**：如何有效建模和检索多模态轨迹数据？需要解决三个层面：(1) 异构轨迹数据的统一表示；(2) 覆盖时序和语义两类关系的检索任务定义；(3) 高效处理长多模态序列的检索模型。本文选择 GUI 环境作为初始探索领域，因为 Web 自动化应用价值大且已有丰富数据资源。

## 方法详解

### 整体框架

本文贡献分三层：
1. **数据层**：构建 Unified Agent Trajectory Dataset (UATD)，整合 5 个开源 GUI 数据源的异构轨迹为统一格式
2. **任务层**：定义 6 类 12 个子任务的多模态轨迹检索基准 GAE-Bench，覆盖时序和语义两类检索关系
3. **模型层**：提出 GAE-Retriever，基于 VLM 的对比学习检索框架，通过 token selection 和 GradCache 解决长多模态序列的内存瓶颈

### 关键设计

1. **UATD 统一轨迹表示**:
    - 功能：将 5 个异构 GUI 数据源（Mind2Web、AutoWebGLM、WebArena、WebLINX、GUIAct）统一为标准轨迹格式
    - 核心思路：将轨迹建模为确定性 MDP $\mathcal{E}=(\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T})$，统一表示为 $\tau = (s_1, a_1, s_2, a_2, \ldots, s_n, a_n)$。状态用原始截图+文本描述表示，动作用操作/目标/值三元组（JSON格式），每个轨迹附带自定义动作空间定义。对缺少截图的数据源（AutoWebGLM）用 gpt-4o-mini 补全 HTML 再用 Playwright 渲染
    - 设计动机：消除平台特定文本表示的依赖，统一格式促进跨环境泛化

2. **GAE-Bench 12 种提取模式**:
    - 功能：从单条轨迹系统性地提取 6 类检索对，形成完整的多模态轨迹检索基准
    - 核心思路：定义时序检索（给前半段检索后半段及其逆、跨粒度的轨迹-状态检索）和语义检索（q→gold trajectory、q→silver trajectory、q→state）。Silver trajectory 通过三步生成：NER 识别实体→生成替代表达→重写查询。总计 714,628 个正样本对，GAE-Bench-lite 限制轨迹长度 ≤10 步含 563,900 对
    - 设计动机：时序检索捕捉轨迹内序列关系，语义检索捕捉跨轨迹功能相似性，12 种模式全面覆盖不同粒度（状态/轨迹/子轨迹）和方向

3. **GAE-Retriever 高效多模态检索**:
    - 功能：基于 VLM2Vec + Qwen2-VL 构建轨迹检索模型，解决多高分辨率截图序列的内存和计算瓶颈
    - 核心思路：**Token Selection**——在 RGB 空间构建 UI 连接图，按相似性聚类后跳过冗余视觉 token，训练时 mask ratio=0.5；**GradCache**——梯度缓存将编码器反传和对比损失反传解耦，支持 sub-batch=1 + 累积 batch=2,048 的大规模对比学习。使用 InfoNCE loss：$\mathcal{L} = -\log \frac{\exp(f(\mathbf{k})^T f(\mathbf{v}^+) / t)}{\sum_{\mathbf{v} \in \mathcal{B}} \exp(f(\mathbf{k})^T f(\mathbf{v}) / t)}$
    - 设计动机：轨迹数据包含多张高分辨率截图，直接编码会 token 数爆炸；对比学习依赖大 batch 的 in-batch negatives，GradCache 突破 GPU 内存限制

### 损失函数 / 训练策略

基于 Qwen2-VL-2B-Instruct，使用 LoRA (rank=8) 在 16 张 H800 GPU 上训练 256 步，总计 1,044 GPU 小时。学习率 $5 \times 10^{-5}$，5% warm-up ratio，最大 token 长度 65,536。Token selection 仅在训练时启用（不引入额外可学习参数），评估时关闭。评估使用 8 张 H800 GPU，batch size=6，耗时 22.5 GPU 小时。

## 实验关键数据

### 主实验（Recall@1/5/10，5 个数据源）

| 方法 | Mind2Web R@1/5/10 | AutoWebGLM R@1/5/10 | WebArena R@1/5/10 | WebLINX R@1/5/10 | GUIAct R@1/5/10 |
|------|-------------------|---------------------|-------------------|------------------|-----------------|
| Qwen2-VL-2B | 0.7/14.5/18.2 | 1.2/6.3/10.7 | 1.4/8.8/12.2 | 3.1/14.2/18.0 | 3.1/8.1/9.4 |
| ColQwen2-v1.0 | 3.2/22.0/29.9 | 3.9/17.7/26.3 | 2.9/13.7/20.0 | 4.2/19.6/25.1 | 6.2/15.5/19.2 |
| GME-Qwen2VL-2B | 3.7/24.2/33.4 | 8.7/27.9/37.4 | 4.2/17.7/24.7 | 5.2/22.4/29.7 | 6.0/16.7/20.7 |
| VLM2Vec-V2.2 | 10.2/44.0/60.1 | 15.7/51.2/67.1 | 9.1/29.1/37.8 | 10.7/38.4/50.5 | 12.2/33.1/40.6 |
| ShowUI-2B | 1.0/13.3/17.0 | 0.8/6.0/8.2 | 1.6/8.5/11.7 | 3.3/13.7/17.3 | 3.1/7.9/9.2 |
| **GAE-Retriever** | **15.0/50.7/67.6** | **22.1/63.6/76.3** | **10.3/31.7/44.1** | **13.7/41.7/54.1** | **25.7/59.2/67.9** |

### 消融实验（与最强基线 VLM2Vec-V2.2 的对比）

| 数据源 | R@1 提升 | R@5 提升 | R@10 提升 |
|--------|---------|---------|----------|
| Mind2Web | +4.8 | +6.7 | +7.5 |
| AutoWebGLM | +6.4 | +12.4 | +9.2 |
| WebArena | +1.2 | +2.6 | +6.3 |
| WebLINX | +3.0 | +3.3 | +3.6 |
| GUIAct | +13.5 | +26.1 | +27.3 |
| **平均** | **+5.8** | **+10.2** | **+10.8** |

### 关键发现

- GAE-Retriever 在全部 5 个数据源的 R@1/5/10 上均最优，最大提升在 GUIAct（R@1 +13.5, R@10 +27.3）
- 多模态 backbone 模型（Qwen2-VL/Qwen2.5-VL）的检索能力极弱（R@1<4.0），说明检索需要专门训练
- Qwen2.5-VL-3B 反而弱于更小的 Qwen2-VL-2B，表明模型规模不等于检索能力
- 轨迹规划模型（ShowUI、UI-TARS、TongUI）的检索能力与 backbone 无显著差异，planning 能力不迁移到 retrieval
- VLM2Vec 系列一致优于其他检索模型，说明融合模态数据+交叉 batch 训练对检索至关重要
- GAE-Retriever 在 OOD 设置下某些任务甚至优于 IND，展示了强泛化能力
- 语义检索任务（q→τ, q→s）相对简单，时序检索（trajectory→trajectory）难度最大

## 亮点与洞察

- **开创性任务定义**：首次系统定义"多模态轨迹检索"任务，建立完整的数据-基准-方法体系，为这一新兴领域奠基
- **12 种提取模式全面覆盖**：时序+语义、6 类检索方向、3 种粒度（状态/轨迹/子轨迹），任务定义的系统性远超同期工作
- **VLM >> CLIP 的实证**：VLM 在处理任意长度多模态输入上天然优于 CLIP-based 模型，截图特定检索模型（UniSE-MLLM）反而最弱
- **Token Selection + GradCache**：训练 trick 有效平衡了高分辨率多截图处理与 GPU 内存限制
- **实用价值高**：框架可直接支持 in-context learning、世界模型、轨迹回放等下游 agent 应用

## 局限与展望

- 仅在 GUI 环境验证，embodied/机器人场景的轨迹检索待探索
- 依赖预训练 VLM 的视觉理解能力，对 GUI 之外的视觉观察可能需要适配
- Silver trajectory 的自动生成质量影响语义检索基准准确性
- GAE-Bench-lite 的轨迹长度限制（≤10 步）可能不代表更长轨迹场景的检索难度
- 检索结果如何集成到下游 agent 决策系统中尚未深入探讨
- 训练资源需求高（16×H800 GPU, 1,044 GPU 小时），复现门槛不低

## 相关工作与启发

- **VLM2Vec / VLM2Vec-V2** (Jiang et al., 2025; Meng et al., 2025)：VLM 用于通用检索的核心 backbone
- **Mind2Web** (Deng et al., 2023)：Web 导航基准，UATD 的数据来源之一
- **AGUVIS** (Xu et al., 2024)：统一 GUI 视觉代理表示，影响了状态表示设计
- **ShowUI** (Lin et al., 2024)：统一动作空间定义，token selection 的灵感来源
- **GradCache** (Gao et al., 2021)：梯度缓存方法，解耦编码器和对比损失的反向传播
- 启发：将"轨迹"作为一等公民进行表示学习和检索是 agent 智能的重要基础设施

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 开创性任务定义+完整数据-基准-方法体系，12种提取模式覆盖全面
- 实验充分度: ⭐⭐⭐⭐ 5个数据源×13个基线方法的全面对比，per-task分析详尽
- 写作质量: ⭐⭐⭐⭐ 结构化清晰，数据集和基准描述详尽，形式语法定义严谨
- 价值: ⭐⭐⭐⭐⭐ 为agent trajectory研究奠定基础设施，数据集和基准有长期使用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](../../ACL2025/multimodal_vlm/megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)
- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](../../NeurIPS2025/multimodal_vlm/generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[ICLR 2026\] U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning](../../ICLR2026/multimodal_vlm/u-marvel_unveiling_key_factors_for_universal_multimodal_retrieval_via_embedding_.md)
- [\[CVPR 2026\] Illuminating Visual Identity in Universal Multimodal Embeddings](../../CVPR2026/multimodal_vlm/illuminating_visual_identity_in_universal_multimodal_embeddings.md)
- [\[ICCV 2025\] GenDoP: Auto-regressive Camera Trajectory Generation as a Director of Photography](../../ICCV2025/multimodal_vlm/gendop_auto-regressive_camera_trajectory_generation_as_a_director_of_photography.md)

</div>

<!-- RELATED:END -->
