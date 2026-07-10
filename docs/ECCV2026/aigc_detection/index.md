---
title: >-
  ECCV2026 AIGC检测论文汇总 · 3篇论文解读
description: >-
  3篇ECCV2026的 AIGC 检测方向论文解读，涵盖模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2026"
  - "AIGC 检测"
  - "论文解读"
  - "论文笔记"
  - "模型压缩"
item_list:
  - u: "diffnet_document_tampering_localization/"
    t: "Efficient Document Tampering Localization with Multi-Level Discrepancy Features and Unified DCT-Quantization Embedding"
  - u: "multi_task_bayesian_in_context_learning/"
    t: "Multi-Task Bayesian In-Context Learning"
  - u: "trustworthy_image_authentication_using_forensic_knowledge_graphs/"
    t: "Trustworthy Image Authentication using Forensic Knowledge Graphs"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**🎞️ ECCV2026** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (10)](../../CVPR2026/aigc_detection/index.md) · [🔬 ICLR2026 (30)](../../ICLR2026/aigc_detection/index.md) · [💬 ACL2026 (17)](../../ACL2026/aigc_detection/index.md) · [🧪 ICML2026 (11)](../../ICML2026/aigc_detection/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/aigc_detection/index.md) · [🧠 NeurIPS2025 (9)](../../NeurIPS2025/aigc_detection/index.md)

**[Efficient Document Tampering Localization with Multi-Level Discrepancy Features and Unified DCT-Quantization Embedding](diffnet_document_tampering_localization.md)**

:   DiffNet 通过两项互补设计——多层差异变换将特征金字塔从内容信号转为符号不变的差异强度信号，以及统一 DCT-量化联合嵌入用离散 embedding 替代传统高开销频率感知头（FPH）——在跨域和人工篡改文档定位上以更低计算成本取得了约 30% 的 F1 提升，吞吐量最高提升 7 倍。

**[Multi-Task Bayesian In-Context Learning](multi_task_bayesian_in_context_learning.md)**

:   本文提出多任务贝叶斯上下文学习（Multi-Task Bayesian ICL），将先验信息通过辅助数据集前缀的形式嵌入到上下文学习序列中，使Transformer能在推理时根据前缀可控地调整后验预测分布，无需重新训练即可在不同先验族间自适应迁移。

**[Trustworthy Image Authentication using Forensic Knowledge Graphs](trustworthy_image_authentication_using_forensic_knowledge_graphs.md)**

:   把图像取证的「找证据」和 VLM 的「说人话」拼成一套系统：先用自监督学到的取证指纹把图像切成来源一致的区域、判定每块区域的源/后处理/压缩历史，组织成一张结构化的「取证知识图谱」，再让 VLM 依图逐条给出可核验的自然语言判据，在检测、伪造类型/定位、取证解释三个维度全面超过纯取证器和纯 VLM。
