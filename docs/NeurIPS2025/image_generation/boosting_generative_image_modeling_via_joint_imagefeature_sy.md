# Boosting Generative Image Modeling via Joint Image-Feature Synthesis

## 基本信息
- **arXiv**: 2504.16064
- **会议**: NeurIPS 2025 **Spotlight**
- **作者**: Theodoros Kouzelis, Efstathios Karypidis, Ioannis Kakogeorgiou, Spyros Gidaris, Nikos Komodakis
- **机构**: University of Crete, Valeo AI
- **代码**: https://representationdiffusion.github.io/

## 一句话总结
提出 Latent-Semantic Diffusion，让扩散模型联合生成 VAE 低级图像 latent 和 DINO 高级语义特征，通过最小修改标准 DiT 实现生成质量和训练效率的显著提升，并解锁 Representation Guidance 推理策略。

## 背景与动机
表征学习和生成建模长期分离：
- **生成模型** (LDM/DiT)：擅长生成高质量图像，但内部表征缺乏语义
- **自监督模型** (DINO/CLIP)：学到强语义表征，但不具备生成能力

将两者结合的尝试通常需要复杂的蒸馏目标或大幅修改架构。能否优雅地在一个模型中同时做到？

## 核心问题
如何让扩散模型在生成图像的同时也生成语义特征，且两者相互增益？

## 方法详解

### 1. 联合图像-特征生成
核心思想：在扩散过程中同时建模两种 latent：
- **Image latent** $z_{img}$：来自预训练 VAE 的低级图像编码
- **Semantic feature** $z_{sem}$：来自预训练 DINO 编码器的高级语义特征

将两者拼接为联合表示，在同一个扩散过程中从纯噪声生成。

### 2. 最小架构修改
- 基于标准 **Diffusion Transformer (DiT)** 架构
- 仅需修改输入/输出投影层以适配拼接后的维度
- 无需复杂的蒸馏损失、对比学习或特殊训练策略
- 统一去噪目标即可同时学习图像和语义的生成

### 3. Representation Guidance（推理时策略）
类比 Classifier-Free Guidance (CFG) 的思路：
- 训练完成后，利用学到的语义特征分支在推理时引导图像生成
- 语义特征→提供高层结构和语义约束→引导低级图像细节生成
- 不需要外部 classifier 或额外模型，自包含的引导机制

### 4. 统一训练
- 相同的去噪损失同时优化图像和语义生成
- 两个分支共享 transformer layers
- 语义分支的引入为图像生成提供隐式正则化
- 训练收敛更快

## 实验关键数据

### 生成质量 (ImageNet)
- 在条件生成和无条件生成设置中均显著提升 FID/IS
- 相比标准 DiT，同参数下实现更低 FID
- **Spotlight 级别**的 improvement

### 训练效率
- 收敛速度显著快于标准 DiT baseline
- 语义分支天然引导模型更快学到正确结构
- 减少了训练迭代次数

### Representation Guidance 效果
- 不使用外部 classifier 就能引导生成
- 与 CFG 互补，可叠加使用
- 生成图像的语义一致性更强

## 亮点
1. **优雅简洁 (Spotlight)**：最小修改 DiT 就能同时生成图像和语义，简约之美
2. **消除蒸馏复杂性**：不需要 REPA/RCFG 等方法的对比蒸馏损失
3. **Representation Guidance**：全新的推理时引导方式，自包含且免费
4. **双赢效果**：联合建模不仅不拖累图像生成，反而提升质量和训练效率
5. **桥接两大领域**：表征学习 + 生成建模的自然统一

## 局限性
1. 依赖预训练 DINO 特征的质量
2. 联合生成增加了 token 数量，推理略慢
3. 主要在 ImageNet 上验证，T2I 场景验证不足
4. 语义特征分支的最优设计可能需要更多探索

## 与相关工作的对比
- **vs. REPA (NeurIPS 2024)**：REPA 用对齐损失将 DINO 表征蒸馏到 DiT，需要额外损失；本文直接联合生成，更简洁
- **vs. REPA-E (ICCV 2025)**：REPA-E 扩展了 REPA 到更多层的蒸馏；本文用完全不同的路径（联合建模 vs. 蒸馏）
- **vs. Classifier Guidance/CFG**：CFG 需要有条件/无条件两次前向或外部 classifier；Representation Guidance 利用已学到的语义分支
- **vs. RCG（表征条件生成）**：RCG 生成 DINO 特征再根据特征生成图像（两步）；本文一步联合生成

## 启发与关联
- **表征与生成的统一趋势**：与 Emu3、Show-o 等统一多模态模型的方向一致——单一模型应同时具备理解和生成能力
- **与 Emergence Concepts (SAE) 的关联**：SAE 发现扩散模型内部已学到语义概念，本文主动注入语义可能加速这一过程
- **Self-supervised + generative 的新范式**：不是先训练 SSL 再训练 generation，而是端到端联合训练

## 评分
- 新颖性：★★★★★ — 联合图像-特征生成是全新范式
- 技术深度：★★★★☆ — 方法简洁但洞察深刻
- 实验完整度：★★★★☆ — ImageNet 实验充分，更多场景待验证
- 写作质量：★★★★★ — Spotlight 论文，表述清晰
