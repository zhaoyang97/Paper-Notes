# SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation

**会议**: ICLR 2026
**arXiv**: [2506.00523](https://arxiv.org/abs/2506.00523)
**代码**: [GitHub](https://github.com/XingtongGe/SenseFlow)
**领域**: 图像生成 / 扩散模型蒸馏
**关键词**: distribution matching distillation, flow matching, text-to-image, few-step generation, FLUX

## 一句话总结

提出 SenseFlow，通过隐式分布对齐（IDA）和段内引导（ISG）将分布匹配蒸馏（DMD）扩展到大规模 flow-based 文生图模型（SD 3.5 Large 8B / FLUX.1 dev 12B），实现 4 步高质量图像生成。

## 研究背景与动机

1. **领域现状**：DMD2 在小模型（SD 1.5、SDXL）上已展现出优秀的蒸馏效果，能将多步扩散模型蒸馏为少步生成器。然而大规模 flow-based 文生图模型（如 SD 3.5 Large 8B、FLUX.1 dev 12B）正成为主流，这些模型的蒸馏仍是open problem。

2. **现有痛点**：原版 DMD2 在大模型上面临三个关键问题：(1) **收敛困难**——即使使用 TTUR（两时间尺度更新规则）也无法稳定训练；(2) **时间步采样不优**——均匀或手工选择的粗时间步未考虑教师模型各时间步的去噪重要性差异；(3) **判别器不够通用**——朴素判别器难以适应不同规模和架构的模型。

3. **核心矛盾**：DMD 的 min-max 博弈框架要求 fake distribution 精确追踪 generator distribution（内部最优响应），但在大模型上这一条件极难满足，导致训练振荡不收敛。

4. **本文要解决什么？**：如何将 DMD 框架可靠地扩展到 8B-12B 参数量的 flow-based 文生图模型？

5. **切入角度**：从 DMD 的 min-max 优化分析入手，发现内部最优响应需要 $p_f = p_g$，设计近端更新（IDA）来近似维持这一条件；用 ISG 重新配置时间步的去噪重要性；引入基于 VFM 的更强判别器。

6. **核心idea一句话**：通过隐式分布对齐维持 fake model 与 generator 的一致性，配合段内引导重分配时间步重要性，实现 DMD 在大 flow 模型上的稳定收敛。

## 方法详解

### 整体框架

以 DMD2 为基础框架，生成器 G 接收文本 prompt 和噪声生成图像，通过 DMD 梯度（real score 与 fake score 之差）、VFM 对抗损失和 ISG 联合优化。每次生成器更新后，IDA 将 fake model 近端对齐到 generator。整体流程见 Algorithm 1。

### 关键设计

**设计1：隐式分布对齐（IDA）**
- **做什么**：在每次 generator 更新后，对 fake model 参数做近端更新 $\phi \leftarrow \lambda\phi + (1-\lambda)\theta$
- **核心思路**：DMD 的内部最优响应要求 $p_f(X_t) = p_g(X_t)$。通过 EMA 式参数对齐，维持 fake model 与 generator 的 ε-best response，即 $E_t D_{KL}(p_g(X_t) \| p_f(X_t)) \leq \varepsilon$
- **设计动机**：单靠增大 TTUR 比例在大模型上既昂贵又不稳定。IDA 以极低成本（仅一次参数插值）维持 fake-generator 一致性，使 DMD 在 SD 3.5 Large 上收敛。λ 接近 1 即可

**设计2：段内引导（ISG）**
- **做什么**：在每个粗时间步的"段"内采样中间时间点，利用教师模型构建引导轨迹
- **核心思路**：对粗时间步 $\tau_i$，采样中间点 $t_{mid} \in (\tau_{i-1}, \tau_i)$。教师从 $\tau_i$ 去噪到 $t_{mid}$，generator 从 $t_{mid}$ 继续到 $\tau_{i-1}$，得到目标 $x_{tar}$。同时 generator 直接从 $\tau_i$ 到 $\tau_{i-1}$，用 L2 损失对齐两者
- **设计动机**：教师模型各时间步的重建误差 $\xi(t)$ 非单调且存在局部振荡，均匀时间步采样造成信息浪费。ISG 将段内的细粒度去噪信息聚合到锚点上，使 generator 更好地近似复杂的段内转换

**设计3：基于 VFM 的判别器**
- **做什么**：使用冻结的视觉基础模型（DINOv2 + CLIP）作为判别器骨干
- **核心思路**：VFM 提取多层语义特征，配合可训练的 head blocks 预测 real/fake logits。使用 hinge loss 训练判别器，generator 的对抗损失按时间步信号功率 $\omega(t) = (1-\sigma_t)^2$ 加权
- **设计动机**：预训练 VFM 引入丰富的语义先验，比朴素判别器更擅长捕获图像质量和细粒度结构。时间步加权确保在高噪声步骤更依赖 DMD 梯度，低噪声步骤更依赖 GAN 反馈

### 损失函数 / 训练策略

Generator 总损失：$\mathcal{L}_G = \mathcal{L}_{DMD} + \lambda_G \cdot \mathcal{L}_{adv} + \lambda_{ISG} \cdot \mathcal{L}_{ISG}$
- $\mathcal{L}_{DMD}$：fake score 与 real score 之差引导 generator
- $\mathcal{L}_{adv}$：VFM 判别器的对抗损失，按 $\alpha_t^2$ 加权
- $\mathcal{L}_{ISG}$：段内引导 L2 损失

训练细节：
- 数据：LAION-5B 子集（aesthetic score ≥ 5.0）
- TTUR 比例：5:1 配合 IDA 即可稳定收敛
- 50% 概率使用 backward simulation vs forward diffusion 构造输入
- Logit-normal 时间步采样

## 实验关键数据

### 主实验

**COCO-5K 4步生成结果**

| 模型 | Patch FID-T↓ | CLIP↑ | HPSv2↑ | PickScore↑ | ImageReward↑ | GenEval↑ |
|------|-------------|-------|--------|------------|-------------|---------|
| SDXL Teacher (80步) | — | 0.3293 | 0.2930 | 22.67 | 0.8719 | 0.5461 |
| DMD2-SDXL (4步) | 21.35 | 0.3231 | 0.2896 | 22.49 | 0.7076 | — |
| **SenseFlow-SDXL (4步)** | **17.xx** | **最优** | **最优** | **最优** | **最优** | **最优** |
| SD 3.5 Teacher (80步) | — | — | — | — | — | 0.7140 |
| SD 3.5 Turbo (4步) | baseline | — | — | — | — | — |
| **SenseFlow-SD3.5 (4步)** | **最优** | **最优** | **超越教师** | **超越教师** | **超越教师** | 0.7098 |
| FLUX.1 schnell (4步) | baseline | — | — | — | — | — |
| **SenseFlow-FLUX (4步)** | **最优** | **最优** | **超越教师** | **超越教师** | **超越教师** | — |

SenseFlow 在所有三个教师模型（SDXL、SD 3.5 Large、FLUX.1 dev）上均实现 SOTA 4步蒸馏。

### 消融实验

| 组件 | SD 3.5 FID 收敛情况 | FLUX 收敛情况 |
|------|-------------------|--------------|
| 原版 DMD2 | 不收敛 | 不收敛 |
| + IDA | 收敛 ✓ | 不收敛 |
| + IDA + ISG | 收敛 ✓ | 收敛 ✓ |
| + IDA + ISG + VFM Disc | 最优 ✓ | 最优 ✓ |

IDA 是 SD 3.5 收敛的关键；ISG 是 FLUX 收敛的额外必要条件。

### 关键发现

1. **IDA 是大模型 DMD 收敛的关键**：仅 IDA 就能让 SD 3.5 Large 稳定收敛，配合小 TTUR 比例即可
2. **ISG 进一步改善 FLUX 蒸馏**：教师时间步重建误差非单调，ISG 通过重分配时间步重要性显著帮助
3. **4步超越教师**：在 HPSv2、PickScore、ImageReward 等人类偏好指标上，SenseFlow 4步甚至超越 80步教师模型
4. **通用性强**：同一框架在 SDXL（2.6B）、SD 3.5（8B）、FLUX（12B）三种不同架构、不同参数量的模型上均有效

## 亮点与洞察

- **首次将 DMD 扩展到 12B 参数的 FLUX 模型**：解决了大规模 flow 模型蒸馏的收敛难题
- **IDA 的简洁优雅**：仅用一行参数插值代码就解决了 fake-generator 分布追踪问题，有理论保证（ε-best response）
- **ISG 的精妙设计**：对教师模型时间步重要性的重新分配思路新颖，利用教师细粒度能力增强粗步 generator
- **4步超越教师**：在人类偏好指标上 4 步蒸馏模型可以超过 80 步教师，说明蒸馏可以"去其糟粕取其精华"
- **VFM 判别器的时间步加权**：低噪声时强调 GAN 信号、高噪声时强调 DMD 信号的设计很有道理

## 局限性 / 可改进方向

1. **计算成本**：需要同时维护 generator、fake model、teacher model 和判别器，显存开销大
2. **IDA 的 λ 选择**：虽然 λ 接近 1 通常足够，但最优值可能因模型而异
3. **ISG 中间点采样策略**：当前均匀采样，可考虑自适应选择更有信息量的中间点
4. **仅评估 4 步**：1步或2步蒸馏效果未知
5. **训练数据依赖**：需要高质量 LAION 子集，数据质量对蒸馏效果的影响未深入分析

## 相关工作与启发

- 对 **DMD/DMD2** 的直接扩展和突破：证明 DMD 框架的核心瓶颈在于 fake-generator 分布追踪
- 与 **RayFlow** 的联系：都关注时间步重要性问题，但 ISG 的解决方案更优雅
- 对蒸馏领域的启发：IDA 思想（近端对齐内部分布）可能适用于其他 min-max 蒸馏框架
- 与 **Consistency Models** 的比较：两种正交的蒸馏路线，DMD 系列在大模型上更有优势

## 评分

- **新颖性**: ⭐⭐⭐⭐ — IDA 和 ISG 都有清晰的理论动机和实践价值，但整体是对 DMD2 的增量改进
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个不同规模和架构的模型、多个基准（COCO-5K/GenEval/T2I-CompBench）、充分的消融
- **写作质量**: ⭐⭐⭐⭐ — 问题分析清晰，但方法描述的数学符号较多
- **价值**: ⭐⭐⭐⭐⭐ — 解决了大规模 flow 模型蒸馏的实际瓶颈，对产业应用有直接推动
