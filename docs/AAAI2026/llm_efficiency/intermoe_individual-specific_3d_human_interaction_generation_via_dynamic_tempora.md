# InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE

**会议**: AAAI 2026
**arXiv**: [2511.13488](https://arxiv.org/abs/2511.13488)
**代码**: [GitHub](https://github.com/Lighten001/InterMoE)
**领域**: LLM Efficiency
**关键词**: 人体交互生成, MoE, 动作生成, 扩散模型, 3D运动合成

## 一句话总结
提出 InterMoE，通过 Dynamic Temporal-Selective MoE 架构解决文本驱动的双人 3D 交互运动生成中的个体特征保持和语义忠实度问题：Synergistic Router 融合语义和运动学特征引导路由，Dynamic Temporal Selection 让专家动态选择关键时间帧，在 InterHuman 上 FID 降低 9%、InterX 上降低 22%。

## 研究背景与动机

1. **领域现状**：文本驱动的双人 3D 交互运动生成是虚拟现实、游戏开发等领域的核心任务。现有方法（InterGen、InterMask、TIMotion）已取得一定进展，但在保持个体特征和语义对齐方面仍有明显不足。
2. **现有痛点**：(a) **跨注意力融合导致个体同质化**——InterGen 等用 cross-attention 融合双人特征后再由标准 FFN 统一处理，压制了个体特征差异，导致两个人动作趋同；(b) **特征拼接导致身份混淆**——TIMotion 等直接拼接双人特征联合生成，缺乏显式身份约束，会出现角色互换或位置错误的问题。
3. **核心矛盾**：需要同时建模个体特征独立性和双人交互依赖性——这两个目标在统一网络中天然矛盾。
4. **切入角度**：MoE 天然适合这个问题——不同专家可以专注于不同个体的运动模式，通过路由机制实现差异化分配。
5. **核心 idea**：(a) Synergistic Router 融合文本语义和运动学特征双重引导路由决策；(b) Dynamic Temporal Selection 让每个专家动态选择关键时间帧（而非固定 Top-K），处理非均匀的时间重要性。

## 方法详解

### 整体框架
输入为文本描述，输出为双人 3D 运动序列 $\mathbf{m}_i \in \mathbb{R}^{T \times J \times d}$。Pipeline 分三部分：
1. **Causal-Skeletal VAE**：骨骼图卷积捕获关节内依赖 + 因果卷积建模时间动态，编码单人运动
2. **Cooperative MoE Denoiser**：两个共享权重的扩散去噪器分别处理两个人，通过 Self-Attention（个体内）+ Cross-Attention（个体间）+ MoE Block 交互
3. **InterMoE Block**：Synergistic Router + Dynamic Temporal Selection

### 关键设计

1. **Synergistic Router（协同路由器）**：
   - 做什么：融合两种信号引导路由——运动路由器基于每个人独特的运动学特征计算路由 logits，文本路由器基于语义计算路由 logits，两者加权融合
   - 公式：$\mathbf{R}^{comb}_{e,s,i} = \alpha \mathbf{R}^{motion}_{e,s,i} + (1-\alpha) \mathbf{R}^{text}_e$
   - 关键创新：采用 batch-level 路由——将 batch 内所有样本的时间特征展平为全局 token 池，让路由器感知不同噪声水平的异质性
   - 设计动机：仅用运动特征路由无法保证语义对齐，仅用文本无法区分不同个体的运动特征

2. **Dynamic Temporal Selection（动态时间选择）**：
   - 做什么：让每个专家动态决定处理多少个时间帧（非固定 Top-K）
   - 核心机制：每个专家有一个可学习偏置 $b_e \in (-1, 0)$，通过 sigmoid + 偏置决定选择门控 $\mathbf{M}_{e,s} = \text{sigmoid}(\mathbf{R}^{comb}_{e,s}) + b_e$，$\mathbf{M}_{e,s} > 0$ 则选中
   - 偏置自适应更新：根据实际选择数量与期望数量的差异调整 $b_e$，训练收敛后趋于稳定
   - 设计动机：交互运动中不同时间帧的重要性不均匀——关键帧（如出拳、躲避）需要更多专家关注，而过渡帧不需要。固定容量的 Token-Choice 和 Expert-Choice 都无法处理这种不均匀性

3. **Causal-Skeletal VAE**：
   - 骨骼图卷积提取关节间空间依赖 + 因果卷积保证时间因果性
   - 轻量但高效的运动表述

## 实验关键数据

### 主实验

| 数据集 | 方法 | FID↓ | R-Precision Top-1↑ | MM-Dist↓ |
|--------|------|------|------------|---------|
| InterHuman | InterGen | 5.149 | 0.489 | 3.785 |
| InterHuman | TIMotion | 5.157 | 0.496 | 3.772 |
| InterHuman | **InterMoE** | **4.677** | **0.512** | **3.762** |
| InterX | InterGen | 0.469 | - | - |
| InterX | **InterMoE** | **0.297** | - | - |

FID 降低：InterHuman -9%（从 5.149 降至 4.677），InterX -22%（从 0.469 降至 0.297）。R-Precision Top-1 从 0.489 提升到 0.512，MultiModality 略低于部分方法，但作者指出这是因为优先保证语义忠实度。

### 消融实验（InterHuman）

| 配置 | FID↓ | R-Precision Top-1↑ | MM-Dist↓ |
|------|------|------------|---------|
| Baseline (InterGen + CS-VAE) | 5.251 | 0.489 | 3.771 |
| w/o Motion & Text Router | 4.782 | 0.503 | 3.766 |
| w/o Batch-level Routing | 6.036 | 0.492 | 3.774 |
| w/o Dynamic Selection | 6.242 | 0.498 | 3.772 |
| w/o Temporal-Selective | 5.195 | 0.505 |
| **Full InterMoE** | **4.677** | **0.512** |

### 关键发现
- **Batch-level 路由和 Dynamic Selection 缺一不可**：去掉任一个 FID 都显著退化（6.036 和 6.242），说明全局视角和动态容量对交互生成都很关键
- **协同路由优于单一信号**：仅用运动或文本路由的效果都不如两者融合，融合后 FID 从 4.782 进一步降至 4.677
- **定性对比清晰展示身份保持优势**：在击剑场景中准确区分攻防双方的手部姿势和前后移动；在拔河场景中精确合成握绳姿态和后仰动作；在 10 秒跆拳道场景中保持圆形移动轨迹——竞争方法在这些场景都出现身份混淆或语义偏离
- **Causal-Skeletal VAE 本身就有贡献**：即使不加 MoE（Baseline 行），引入因果-骨骼 VAE 相比原始 InterGen 也有改善
- **在单人运动生成上也有竞争力**：验证了方法的通用性，不限于交互场景

## 亮点与洞察
- 用 MoE 架构解决双人交互中的"个体特征保持"问题是一个自然且优雅的选择——不同专家可以自动专注于不同个体的运动模式或不同动作阶段。Synergistic Router 的双信号融合确保了语义和运动学的同时对齐，避免了"语义正确但运动学不自然"或"动作流畅但语义偏离"的单一优化陷阱
- Dynamic Temporal Selection 通过可学习偏置实现的"弹性容量"机制很实用——不同于固定 Top-K 的刚性选择，偏置的自适应更新让系统在训练过程中自动发现最优容量分配。这个设计也可以推广到视频生成等其他有时间非均匀重要性的任务
- Batch-level 路由策略值得关注——让路由器能感知整个 batch 中不同噪声水平的样本差异，这是扩散模型 MoE 的一个关键设计考量

## 局限性 / 可改进方向
- 仅在双人交互上验证，能否扩展到多人（3+）交互场景未知——多人交互的组合复杂度会急剧增加
- Synergistic Router 的融合权重 $\alpha=0.5$ 是固定的，可以学习自适应权重，让模型根据任务自动调整语义和运动学信号的相对重要性
- 评价指标（FID、R-Precision）可能无法完全反映个体特征保持的质量——需要设计更针对性的身份一致性指标（如计算同一角色在序列中的动作风格一致性）
- 训练在两张 RTX3090 上完成，计算成本合理，但 batch-level routing 在超大 batch 下可能面临内存瓶颈

## 相关工作与启发
- **vs InterGen**：用 cross-attention 做交互但后续 FFN 统一处理导致同质化。InterMoE 用 MoE 替代 FFN，不同专家处理不同模式，避免同质化
- **vs TIMotion**：拼接双人特征联合生成，缺少身份约束。InterMoE 的两个 Cooperative Denoiser 分别处理各自个体，天然保持身份
- **vs EC-DiT/DiT-MoE**：通用的扩散 MoE 方法，InterMoE 针对交互运动的时间非均匀性设计了动态时间选择
- **vs ComMDM**：用小网络桥接两个单人扩散模型，但在有限交互数据集上训练效果受限。InterMoE 的 Cooperative Denoiser 设计更深入地建模了双人依赖
- **vs in2IN**：引入个体行动描述作为额外条件，但仍使用统一 FFN 处理。InterMoE 的 MoE 替代 FFN 从架构层面保证差异化处理

## 评分
- 新颖性: ⭐⭐⭐⭐ MoE 用于交互运动生成是新尝试，Synergistic Router 和 Dynamic Temporal Selection 设计有针对性
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、定量+定性对比、详细消融分析
- 写作质量: ⭐⭐⭐⭐ 方法每个组件的动机都有清晰说明
- 价值: ⭐⭐⭐⭐ 双人交互运动生成的新 SOTA，FID 降低 9-22% 的实质性改进
