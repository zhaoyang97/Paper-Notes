---
title: >-
  ICML2026 地球科学论文汇总 · 2篇论文解读
description: >-
  2篇ICML2026的地球科学方向论文解读，涵盖时序预测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "地球科学"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
item_list:
  - u: "scaling_laws_of_global_weather_models/"
    t: "Scaling Laws of Global Weather Models"
  - u: "sparse_attention_to_the_details_preserving_spectral_fidelity_in_ml-based_weather/"
    t: "(Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌍 地球科学

**🧪 ICML2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/earth_science/index.md) · [🔬 ICLR2026 (7)](../../ICLR2026/earth_science/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/earth_science/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/earth_science/index.md) · [📷 CVPR2025 (1)](../../CVPR2025/earth_science/index.md) · [🎞️ ECCV2024 (1)](../../ECCV2024/earth_science/index.md)

**[Scaling Laws of Global Weather Models](scaling_laws_of_global_weather_models.md)**

:   本文在统一的训练/评测协议下，对 5 个主流数据驱动天气模型（Aurora、AIFS、Pangu、GraphCast、SFNO）做了首个跨模型的缩放定律分析，发现天气模型偏爱"加宽而非加深"、计算预算应优先投给更多训练数据而非更大模型，且不同气象变量的缩放行为差异巨大——这些规律与 NLP/视觉的缩放定律截然不同。

**[(Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models](sparse_attention_to_the_details_preserving_spectral_fidelity_in_ml-based_weather.md)**

:   MOSAIC 用"概率扰动 + 在 HEALPix 球面网格上的 mesh-aligned 块稀疏注意力"同时解决了 ML 天气预报模型的两类频谱退化（确定性平均带来的谱衰减 + 粗化潜空间带来的高频走样），在 1.5° 分辨率上仅 214M 参数就匹敌甚至超过 6× 高分辨率的模型，单 H100 12 秒生成 24 成员 10 天预报。
