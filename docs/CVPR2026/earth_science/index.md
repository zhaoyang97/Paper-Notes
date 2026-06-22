---
title: >-
  CVPR2026 地球科学论文汇总 · 2篇论文解读
description: >-
  2篇CVPR2026的地球科学方向论文解读，涵盖扩散模型、时序预测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "地球科学"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "时序预测"
item_list:
  - u: "phyoceancast_global_ocean_forecasting_with_physics-informed_diffusion/"
    t: "PhyOceanCast: Global Ocean Forecasting with Physics-Informed Diffusion"
  - u: "sigma_a_physics-based_benchmark_for_gas_chimney_understanding_in_seismic_images/"
    t: "SIGMA: A Physics-Based Benchmark for Gas Chimney Understanding in Seismic Images"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌍 地球科学

**📷 CVPR2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (7)](../../ICLR2026/earth_science/index.md) · [🧪 ICML2026 (2)](../../ICML2026/earth_science/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/earth_science/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/earth_science/index.md) · [📷 CVPR2025 (1)](../../CVPR2025/earth_science/index.md) · [🎞️ ECCV2024 (1)](../../ECCV2024/earth_science/index.md)

**[PhyOceanCast: Global Ocean Forecasting with Physics-Informed Diffusion](phyoceancast_global_ocean_forecasting_with_physics-informed_diffusion.md)**

:   PhyOceanCast 把全球海洋预报建模成一个**残差扩散**问题，用球面图注意力网络（SGAN-MOC）解决"高纬投影畸变 + 变量耦合"、用物理小波时序模块（PWTC）解决"多尺度动力学 + 守恒约束"，一次预报 145 个海洋变量、36 个深度层，30 天预报 RMSE 相对最优 baseline 降低约 13.7%。

**[SIGMA: A Physics-Based Benchmark for Gas Chimney Understanding in Seismic Images](sigma_a_physics-based_benchmark_for_gas_chimney_understanding_in_seismic_images.md)**

:   本文提出首个带真值标注的物理合成地震图像数据集 SIGMA——用波动方程正演+逆时偏移把含气烟囱的速度模型转成地震图像，同时给出像素级气烟囱掩码（用于检测）和"退化—干净"配对图（用于增强），并在两类任务上 benchmark 多个基线，揭示现有方法在该数据上集体吃力。
