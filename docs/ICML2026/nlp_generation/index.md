---
title: >-
  ICML2026 文本生成论文汇总 · 2篇论文解读
description: >-
  2篇ICML2026的文本生成方向论文解读，收录 Characterizing the Effect of N、Score-Repellent Monte Carlo等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "文本生成"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "characterizing_the_effect_of_noise_in_language_generation_in_the_limit/"
    t: "Characterizing the Effect of Noise in Language Generation in the Limit"
  - u: "score-repellent_monte_carlo_toward_efficient_non-markovian_sampler_with_constant/"
    t: "Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✍️ 文本生成

**🧪 ICML2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (12)](../../ICLR2026/nlp_generation/index.md) · [💬 ACL2026 (17)](../../ACL2026/nlp_generation/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/nlp_generation/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/nlp_generation/index.md) · [🧪 ICML2025 (1)](../../ICML2025/nlp_generation/index.md) · [💬 ACL2025 (27)](../../ACL2025/nlp_generation/index.md)

**[Characterizing the Effect of Noise in Language Generation in the Limit](characterizing_the_effect_of_noise_in_language_generation_in_the_limit.md)**

:   在 Kleinberg-Mullainathan 的"语言极限生成"形式化框架下，本文证明了对于均匀和非均匀生成，噪声水平 1 与任意有限噪声水平 $i \geq 1$ 等价（层级坍缩），但无噪声与噪声 1 之间存在严格分离，并首次给出了非均匀噪声依赖可生成性的完整刻画。

**[Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces](score-repellent_monte_carlo_toward_efficient_non-markovian_sampler_with_constant.md)**

:   SRMC 用一个 $d$ 维的 running score 平均（而不是 $|\mathcal{X}|$ 维的经验测度）来记录历史，再通过指数 score-tilt 把这段历史折成一个"排斥已访问区域"的代理目标 $\pi_\theta$，套在任何 base MCMC kernel 外面，就能在通用状态空间下用常数内存实现非马尔可夫、低方差、保持归一化无关性的采样器。
