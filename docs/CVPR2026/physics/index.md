---
title: >-
  CVPR2026 物理/科学计算论文汇总 · 2篇论文解读
description: >-
  2篇CVPR2026的物理/科学计算方向论文解读，涵盖时序预测、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "物理/科学计算"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "扩散模型"
item_list:
  - u: "aviasafe_a_physics-informed_data-driven_model_for_aviation_safety-critical_cloud/"
    t: "AviaSafe: A Physics-Informed Data-Driven Model for Aviation Safety-Critical Cloud Forecasts"
  - u: "spatial-spectral_residuals_informed_diffusion_neural_operator_for_pan-sharpening/"
    t: "Spatial-Spectral Residuals Informed Diffusion Neural Operator for Pan-sharpening"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚛️ 物理/科学计算

**📷 CVPR2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (69)](../../ICLR2026/physics/index.md) · [🧪 ICML2026 (33)](../../ICML2026/physics/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/physics/index.md) · [🧠 NeurIPS2025 (57)](../../NeurIPS2025/physics/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/physics/index.md) · [🧪 ICML2025 (20)](../../ICML2025/physics/index.md)

**[AviaSafe: A Physics-Informed Data-Driven Model for Aviation Safety-Critical Cloud Forecasts](aviasafe_a_physics-informed_data-driven_model_for_aviation_safety-critical_cloud.md)**

:   AviaSafe 把"先用掩码定位云在哪、再回归云有多浓"的层级化思路和航空气象里验证多年的"结冰条件指数(IC)"嵌进一个 Swin Transformer 预报骨干里，第一次实现了全球、逐 6 小时、可分相态（冰/液/雨/雪）的云微物理量预报，在 93.7% 的变量-时效组合上优于 FuXi 基线，并在 7 天时效的关键背景变量上追平甚至超过业务级数值预报 ECMWF HRES。

**[Spatial-Spectral Residuals Informed Diffusion Neural Operator for Pan-sharpening](spatial-spectral_residuals_informed_diffusion_neural_operator_for_pan-sharpening.md)**

:   SRINO 把全色锐化的扩散去噪骨干从注意力换成 Galerkin 型神经算子（把生成过程搬到连续函数空间、显著省 FLOPs 和显存），再在每一步反向采样里直接把像素级的空间/光谱一致性残差当条件喂进去做闭环引导，在 WV3/GF2/QB 三个数据集上既超过现有 SOTA 又比注意力扩散省好几倍算力。
