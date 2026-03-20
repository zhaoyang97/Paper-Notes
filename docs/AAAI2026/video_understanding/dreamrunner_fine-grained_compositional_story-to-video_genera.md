# DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation

**会议**: AAAI 2026  
**arXiv**: [2411.16657](https://arxiv.org/abs/2411.16657)  
**代码**: 未公开  
**领域**: 视频生成  
**作者**: Zun Wang, Jialu Li, Han Lin, Jaehong Yoon, Mohit Bansal (UNC Chapel Hill)
**关键词**: 故事视频生成, 检索增强运动适配, 区域注意力, 组合式视频生成, LoRA注入  

## 一句话总结

提出 DreamRunner 框架，通过 LLM 双层规划 + 检索增强运动先验学习 + 时空区域3D注意力模块(SR3AI)，实现细粒度可控的多角色多事件故事视频生成。

## 背景与动机

故事视频生成 (SVG) 旨在根据叙事脚本生成多场景、角色驱动的连贯视频。现有方法（如 VLogger、VideoDirectorGPT）主要用 LLM 做高层场景分解，各场景独立生成后拼接，存在三个核心挑战：

1. **连贯组合**：单场景描述往往包含多个对象/角色的不同运动轨迹、属性和顺序事件，难以连贯组合
2. **复杂运动合成**：复杂场景描述涉及精细角色运动（如"交谊舞"），基础 T2V 模型难以直接生成
3. **角色定制 + 顺序事件**：需同时保持角色视觉一致性和时序运动连贯性

现有 SVG 方法直接将场景描述作为文本条件输入 T2V 模型，约束有限，导致保真度差、事件/对象丢失、运动模糊。

## 核心问题

如何在故事视频生成中实现：(1) 多对象多事件的细粒度组合控制；(2) 基于检索的复杂运动定制化合成；(3) 多角色一致性与顺序运动的无冲突注入？

## 方法详解

### 整体框架

DreamRunner 包含三个阶段：

**阶段1：双层视频规划生成 (Dual-Level Plan Generation)**
- **故事级粗粒度规划**：用 GPT-4o 从故事主题生成 6~8 个角色驱动、运动丰富的场景描述，每个描述包含 scene、motions、narrations 三个结构化字段
- **场景级细粒度规划**：将每个场景描述分解为逐帧(6个关键帧)的实体级布局规划，格式为 `Frame: [entity name, entity motion, entity description], [x0,y0,x1,y1]`，坐标归一化到 [0,1]。重叠区域生成合并描述

**阶段2：运动检索与先验学习 (Motion Retrieval & Prior Learning)**
- **运动视频检索**：从 InternVid 大规模视频数据库中检索运动相关视频
  - BM25 初步文本检索（400个候选）→ 属性过滤（时长≥2s、帧数≥40、宽高比≥0.9）→ YOLOv5 目标追踪裁剪 → CLIP+ViCLIP 语义相似度排序
  - 每个运动检索 4~20 个视频片段
- **运动先验训练**：基于 MotionDirector 的 test-time fine-tuning
  - 在 CogVideoX 的 3D full attention 上手动指定偶数层为"空间层"、奇数层为"时间层"
  - 空间 LoRA 学外观，时间 LoRA 学运动
  - **关键创新**：使用 per-video prompt（而非所有视频共享单一 prompt），帮助模型隔离运动无关的背景/外观
- **角色先验学习**：将参考图像重复48次构造伪视频，LoRA 注入空间层，仅重建首帧防止过拟合

**阶段3：时空区域扩散 (SR3AI)**

### 关键设计

**SR3AI (Spatial-Temporal Region-Based 3D Attention and Prior Injection)**

核心是对 CogVideoX 的 3D full attention 进行区域化改造：

1. **区域3D注意力**：
   - 给定 N 个区域文本描述 $C_1,...,C_N$ 和对应布局 $L_1,...,L_N$
   - 编码每个文本条件得到嵌入 $T_1,...,T_N$
   - 注意力掩码规则：每个区域视觉 latent $L_i$ 可以注意到其对应文本 $T_i$ 及所有视觉 latent $L_1,...,L_N$；每个文本嵌入 $T_i$ 只注意自身和对应 $L_i$
   - 保证各区域受各自文本约束，同时视觉 latent 间保持交互（非硬隔离）

2. **区域 LoRA 注入**：
   - 对每个 LoRA 根据文本描述和布局信息计算 latent mask
   - 公式：$Wx = W_0x + A_{\text{witch}}B_{\text{witch}}(Mask_{\text{witch}} \cdot x) + A_{\text{cat}}B_{\text{cat}}(Mask_{\text{cat}} \cdot x)$
   - 角色 LoRA 注入空间层，运动 LoRA 注入时间层，无层级重叠 → 避免多 LoRA 冲突

### 损失函数 / 训练策略

运动先验训练使用两个损失：

- **标准扩散损失** $L_{org}$：重建所有视频帧
$$L_{org} = \mathbb{E}_{z_0, y, \epsilon \sim \mathcal{N}(0,1), t \sim U(0,T)} [\|\epsilon - \epsilon_\theta(z_t, t, y)\|^2]$$

- **外观去偏时序损失** $L_{ad}$：在归一化 latent 空间中聚焦运动学习
$$\phi(\epsilon) = \frac{\epsilon}{\sqrt{\beta^2 + 1}} - \beta \cdot \epsilon_{anchor}$$
$$L_{ad} = \mathbb{E}[\|\phi(\epsilon) - \phi(\epsilon_\theta(z_t, t, y))\|^2]$$

- 总损失 $L_{motion} = L_{org} + L_{ad}$（无需额外权重调节，简单鲁棒）

角色先验：LoRA 注入空间层，仅重建首帧。无文本条件 fine-tuning，每个先验约 5min/A6000。

## 实验关键数据

**故事视频生成 (DreamStorySet, Table 1)**

| 指标 | VideoDirectorGPT | VLogger | DreamRunner | 相对提升 |
|------|:-:|:-:|:-:|:-:|
| Character CLIP ↑ | 54.3 | 62.5 | **70.7** | +13.1% |
| Character DINO ↑ | 9.5 | 41.3 | **55.1** | +33.4% |
| Fine-Grained CLIP ↑ | 23.7 | 23.5 | **24.7** | +5.1% |
| Fine-Grained ViCLIP ↑ | 21.7 | 23.1 | **23.7** | +2.6% |
| Full Text CLIP ↑ | 22.4 | 22.5 | **24.2** | +7.6% |
| Full Text ViCLIP ↑ | 22.5 | 22.2 | **24.1** | +8.6% |
| Transition DINO ↑ | 63.5 | 73.6 | **93.6** | +27.2% |
| Aesthetics ↑ | 42.3 | 43.4 | **55.4** | +27.6% |
| Imaging ↑ | 60.3 | 61.2 | **62.1** | +1.5% |
| Smoothness ↑ | 94.3 | 96.2 | **98.1** | +2.0% |

**组合式 T2V (T2V-CompBench, Table 2)**

SR3A 模块（无 LoRA 注入版本）应用到 CogVideoX-2B/5B 上：
- Dynamic attribute binding 提升超过 25%
- Spatial binding 提升超过 15%
- Motion binding 提升至少 10%
- CogVideoX-5B + SR3A 在开源模型中 5 个维度达到 SoTA
- 在 dynamic attribute binding、spatial binding、object interactions 维度超越所有闭源模型（Gen-3、Dreamina、PixVerse、Kling）

**RAG + Per-Video Prompt 效果 (Table 4)**

| 方法 | CLIP | ViCLIP |
|------|:-:|:-:|
| CogVideoX-2B 基线 | 23.39 | 20.84 |
| + RAG (单一prompt) | 24.01 | 22.02 |
| + RAG (per-video prompt) | **24.67** | **23.04** |

### 消融实验要点

**SR3AI + RAG 组合效果 (Table 3)**：
- 仅 SR3AI → 显著改善事件转换平滑度 + 视觉质量 + 文本对齐（分而治之效应）
- 仅 RAG → 改善细粒度和全文本的视频-文本相似度
- 两者结合 → 所有维度最优

**RAG Pipeline 消融 (Table 5)**：
- 最大检索数 20 + CLIP/ViCLIP 过滤 → CLIP 25.47, ViCLIP 23.66（最佳）
- 仅 3 个视频 + 过滤 → CLIP 24.45, ViCLIP 22.80（不够）
- 20 个不过滤 → CLIP 24.01, ViCLIP 22.51（有噪声但仍有效）

**层分离策略 (Table 6)**：
- 交错注入（奇偶层）> 前后半注入 > 无外观去偏
- 交错注入：CLIP 25.5, ViCLIP 23.7

**视觉质量 (Table 7-8)**：
- 加入 RAG 和 SR3AI 不损害视觉质量，反而全面提升
- 总体质量分：82.55（DreamRunner） vs 78.6（VLogger） vs 75.3（VideoDirectorGPT）

**计算成本**：
- TTF ~0.2 GPU hours/运动，全故事 3~4 GPU hours
- 对比：VLogger ~6K GPU hours，VideoDirectorGPT ~400 GPU hours

## 亮点

1. **检索增强运动定制**是重要创新：将运动合成转化为定制化问题，从 InternVid 检索相关视频做 test-time fine-tuning，无需人工收集训练数据
2. **Per-video prompt** 设计简单但有效，帮助模型去除外观/背景噪声专注运动模式
3. **SR3AI 的区域化设计**同时实现了时空区域注意力和区域 LoRA 注入，且不需要额外训练，zero-shot 工作
4. **空间/时间 LoRA 天然分层**避免了多 LoRA 冲突，架构设计简洁
5. 在 T2V-CompBench 上部分维度超过闭源模型，证明开源模型 + 好设计可缩小差距
6. 计算量大幅下降（3~4 GPU hours vs 6K），实用性强

## 局限性 / 可改进方向

1. **性能受限于骨干模型**：基于 CogVideoX-2B/5B，如果骨干在罕见组合或复杂运动上能力不足，DreamRunner 也会继承这些弱点
2. **奇偶层分离是 heuristic**：手动指定偶数层为空间、奇数层为时间缺乏理论依据，虽然消融表明有效但对其他架构的泛化性存疑
3. **检索依赖外部数据库**：运动检索依赖 InternVid 数据库的覆盖度，对罕见运动可能检索不到合适的视频
4. **重叠区域处理粗糙**：多角色重叠区域通过 LLM 合并描述处理，质量依赖于 LLM 的合并能力
5. **评测数据集自建且较小**：DreamStorySet 仅 10 个角色、64 个运动，多角色仅做定性评估，评测可能不够充分
6. **每个运动需要单独 TTF**：虽然每个仅 5min，但 15-20 个运动串行执行仍需数小时
7. **6秒视频较短**：每个场景仅生成6秒视频（8fps），对长叙事的表达力有限

## 与相关工作的对比

| 方法 | 多对象控制 | 运动定制 | 时空区域控制 | 角色一致性 | 训练需求 |
|------|:-:|:-:|:-:|:-:|:-:|
| VideoDirectorGPT | 空间布局 | ✗ | 仅空间 | 弱 | 400 GPU hrs |
| VLogger | ✗ | ✗ | ✗ | 中等 | 6K GPU hrs |
| Peekaboo | 区域mask | ✗ | 仅空间 | - | - |
| TALC | ✗ | ✗ | 仅时间 | - | - |
| MotionDirector | ✗ | 单视频FT | ✗ | - | 按视频 |
| **DreamRunner** | **区域3D注意力** | **RAG+FT** | **时空同时** | **强** | **3-4 GPU hrs** |

关键差异：DreamRunner 首次同时实现了时空区域控制（SR3AI）和检索增强运动定制，且通过区域化 LoRA 注入实现无冲突的多角色多运动生成。

## 启发与关联

1. **RAG 范式在生成模型中的应用**值得借鉴：不直接从头训练，而是从大规模数据库检索+轻量适配，在其他条件生成任务（音频、3D）中也可尝试
2. **区域化 LoRA 注入**的思路可推广到任何多概念组合生成场景（如多风格混合图像生成）
3. **Per-video prompt** 的成功暗示现有运动学习方法（共享 prompt）中存在显著的运动-外观耦合问题
4. 与 DiT 架构的 3D full attention 特性紧密结合，在 UNet 架构上不一定能直接复用
5. 未来更强骨干（如 Sora 类模型）+ 类似的细粒度控制模块可能产生质的飞跃

## 评分 ⭐⭐⭐⭐ (4/5)

**优势**：框架完整、设计环环相扣（规划→检索→注入），实验充分（两个任务 + 详尽消融），多个维度大幅超越基线，计算效率高。

**不足**：自建评测集规模小且多角色仅定性评估，奇偶层分离缺乏理论支撑，6秒视频限制了故事表达力，整体仍受限于 CogVideoX 骨干能力。

**总评**：一篇工程性和系统性很强的故事视频生成工作，将 RAG、区域注意力、LoRA 定制三个技术组件有机整合。虽然单个模块的新颖性有限，但组合效果显著且实验扎实。
