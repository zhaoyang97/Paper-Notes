---
title: >-
  ICLR2026 遥感论文汇总 · 5篇论文解读
description: >-
  5篇ICLR2026的遥感方向论文解读，涵盖 Agent、多模态、时序预测、遥感等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "遥感"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "多模态"
  - "时序预测"
item_list:
  - u: "earth-agent_unlocking_the_full_landscape_of_earth_observation_with_agents/"
    t: "Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents"
  - u: "measuring_the_intrinsic_dimension_of_earth_representations/"
    t: "Measuring the Intrinsic Dimension of Earth Representations"
  - u: "spectral_gaps_and_spatial_priors_studying_hyperspectral_downstream_adaptation_us/"
    t: "Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind"
  - u: "tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t/"
    t: "TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models"
  - u: "task-free_adaptive_meta_black-box_optimization/"
    t: "Task-free Adaptive Meta Black-box Optimization"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**🔬 ICLR2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (63)](../../CVPR2026/remote_sensing/index.md) · [🧪 ICML2026 (3)](../../ICML2026/remote_sensing/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/remote_sensing/index.md) · [🧠 NeurIPS2025 (12)](../../NeurIPS2025/remote_sensing/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/remote_sensing/index.md) · [🧪 ICML2025 (7)](../../ICML2025/remote_sensing/index.md)

**[Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents](earth-agent_unlocking_the_full_landscape_of_earth_observation_with_agents.md)**

:   Earth-Agent是首个基于MCP工具生态的地球观测Agent框架，统一了RGB和光谱遥感数据，通过动态调用104个专家工具实现跨模态、多步骤、定量时空推理，配套提出的Earth-Bench基准包含248个专家任务和13,729张图像，实验证明Earth-Agent远超通用Agent和遥感MLLM。

**[Measuring the Intrinsic Dimension of Earth Representations](measuring_the_intrinsic_dimension_of_earth_representations.md)**

:   首次系统度量地理隐式神经表示（Geographic INR）的内在维度（ID），发现256-512维嵌入的真实ID仅2-10维；冻结嵌入空间的高ID与好的下游性能正相关，而监督任务头激活空间的低ID与高性能正相关，揭示了「代表性 vs 任务对齐」的双重机制。

**[Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind](spectral_gaps_and_spatial_priors_studying_hyperspectral_downstream_adaptation_us.md)**

:   研究未经高光谱预训练的多模态地理空间基础模型 TerraMind 能否通过通道适配策略（朴素波段选择 vs. SRF 分组）有效适配高光谱下游任务，结果表明朴素波段选择一致优于物理感知的 SRF 方法，但性能差距随任务光谱复杂度增大而扩大。

**[TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)**

:   提出 TAMMs——首个统一框架，在单一 MLLM-扩散架构中联合执行卫星图像时间序列的时序变化描述（TCD）和未来图像预测（FSIF），通过时序适配模块（TAM）唤醒冻结 MLLM 的时序推理能力，并通过语义融合控制注入（SFCI）机制将变化理解转化为生成控制信号。

**[Task-free Adaptive Meta Black-box Optimization](task-free_adaptive_meta_black-box_optimization.md)**

:   提出 ABOM——一种无需预定义训练任务的自适应元黑盒优化器，通过将进化算子（选择、交叉、变异）参数化为可微注意力模块，在优化过程中利用自生成数据在线更新参数，在合成基准和无人机路径规划上实现零样本竞争性能。
