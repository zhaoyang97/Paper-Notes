---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**📹 ICCV2025** · 共 **2** 篇

**[A Good Teacher Adapts Their Knowledge for Distillation](a_good_teacher_adapts_their_knowledge_for_distillation.md)**

:   本文揭示了知识蒸馏中教师-学生容量差距问题的本质原因在于**输出分布的类内分布不匹配**，并提出 AID（Adapted Intra-class Distribution）方法，在蒸馏前对教师模型进行微调以优化其类内分布使之更符合学生的学习能力，在多种架构组合上取得了SOTA性能。

**[TokenBridge: Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](bridging_continuous_and_discrete_tokens_for_autoregressive_v.md)**

:   TokenBridge提出对预训练VAE连续特征进行后训练维度级量化，将连续token无损转化为离散token，再通过轻量级维度级自回归头高效建模指数级大词表空间，在ImageNet 256×256上用标准交叉熵损失达到了与连续token方法（如MAR）相当的生成质量（FID=1.55），且推理快5.94倍。
