# JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization

**会议**: ICLR 2026  
**arXiv**: [2503.23377](https://arxiv.org/abs/2503.23377)  
**代码**: [https://javisverse.github.io/JavisDiT-page/](https://javisverse.github.io/JavisDiT-page/)  
**领域**: 扩散模型 / 视频生成  
**关键词**: 音视频联合生成, DiT, 时空同步, 对比学习, 基准数据集  

## 一句话总结

提出 JavisDiT，基于 DiT 架构的音视频联合生成模型，通过层级化时空同步先验估计器（HiST-Sypo）实现细粒度的音视频时空对齐；同时构建了新基准 JavisBench（10K 复杂场景样本）和新评估指标 JavisScore。

## 研究背景与动机

1. **音视频联合生成（JAVG）的兴起**：音频和视频在现实场景中天然耦合，联合生成对影视制作和短视频创作有重要价值
2. **异步级联方法的局限**：先生成音频再合成视频（或反之）会累积噪声，端到端方法更有前景
3. **现有 DiT 骨干的空间时序建模不足**：AV-DiT 和 MM-LDM 使用图像 DiT，难以建模精细时空关系
4. **同步对齐策略的粗糙**：现有方法仅实现粗粒度的时间对齐（参数共享）或语义对齐（嵌入对齐），缺乏空间维度的细粒度同步
5. **评估基准的简单性**：AIST++ 和 Landscape 等数据集场景单一，无法反映真实世界的复杂多事件场景
6. **评估指标的缺陷**：AV-Align 依赖光流和音频 onset 检测，在复杂场景下不可靠

## 方法详解

### 整体框架

JavisDiT 包含视频分支和音频分支，共享 AV-DiT blocks。每个分支依次经过：**ST-SelfAttn** → **粗粒度 CrossAttn**（T5 语义） → **细粒度 ST-CrossAttn**（时空先验） → **双向 CrossAttn**（跨模态融合）。

### 关键设计一：层级时空同步先验估计器（HiST-Sypo）

**粗粒度先验**：直接复用 T5 编码器的语义嵌入，描述整体声音事件

**细粒度先验估计**：
- 使用 ImageBind 文本编码器的 77 个隐状态作为输入
- $N_s = 32$ 个空间 token 和 $N_t = 32$ 个时间 token 作为查询
- 4 层 Transformer encoder-decoder $\mathcal{P}$ 提取时空先验
- 输出高斯分布的均值和方差，采样得到随机的时空先验 $(p_s, p_t) \leftarrow \mathcal{P}_\phi(s; \epsilon)$
- 通过 **对比学习** 训练：构造负样本（异步音视频对）和专用损失函数

### 关键设计二：多模态双向交叉注意力（MM-BiCrossAttn）

- 计算视频 $q_v$ 和音频 $k_a$ 的注意力矩阵 $A$
- $A \times v_a$ → 音频到视频注意力
- $A^T \times v_v$ → 视频到音频注意力
- 双向信息流实现跨模态深度融合

### 三阶段训练策略

1. **音频预训练**（0.8M 音频-文本对）：用 OpenSora 的视频分支权重初始化音频分支
2. **ST-Prior 训练**（0.6M 同步音视频三元组）：训练 HiST-Sypo 估计器
3. **JAVG 训练**（0.6M 样本）：冻结 SA 和 ST-Prior，仅训练 ST-CrossAttn 和 Bi-CrossAttn

### 损失函数

- 扩散去噪损失（FlowMatching 或 DDPM）
- ST-Prior 估计器：对比学习损失（同步正样本 vs 异步负样本）
- 动态时间 masking 支持多种条件任务

## 实验关键数据

### JavisBench 主要结果

| 方法 | FVD ↓ | FAD ↓ | TV-IB ↑ | AV-IB ↑ | JavisScore ↑ |
|------|-------|-------|---------|---------|-------------|
| TempoToken (T2A→A2V) | 539.8 | - | 0.084 | - | - |
| MM-Diffusion (JAVG) | - | - | - | - | - |
| **JavisDiT** | **Best** | **Best** | **Best** | **Best** | **Best** |

### JavisBench 数据集特点

| 维度 | 类别数 | 特点 |
|------|--------|------|
| 事件场景 | 多类 | 自然、工业、室内等 |
| 空间组成 | 2 | 单/多发声主体 |
| 时间组成 | 3 | 单事件/顺序/并发 |
| 总样本数 | 10,140 | 75% 含多事件，57% 含并发事件 |

### AIST++ 和 Landscape 对比

JavisDiT 在传统基准（FVD、KVD、FAD 指标）上也显著优于 MM-Diffusion 和级联方法。

## 亮点与洞察

1. **细粒度时空对齐**：不仅对齐"什么时候发声"，还对齐"在画面哪里发声"——这是之前工作忽略的空间维度
2. **随机化先验采样**：同一文本可对应不同的时空先验分布，建模了事件发生位置和时间的不确定性
3. **JavisBench 的挑战性**：75% 样本含多事件，57% 含并发事件，远超现有基准复杂度
4. **JavisScore 的鲁棒性**：分窗口计算 ImageBind 同步分数并选取最不同步的 40% 帧，比 AV-Align 更可靠
5. **模块化设计**：冻结单模态 SA 块，仅训练跨模态模块，参数高效

## 局限性 / 可改进方向

1. 视频生成分辨率较低（240P/24fps），与最新视频模型有差距
2. 依赖 OpenSora 预训练权重，独立训练的可行性未验证
3. ImageBind 的音视频嵌入空间可能在极端场景下不够精细
4. HiST-Sypo 估计器的 $N_s = 32, N_t = 32$ 的设置是否最优未深入探讨
5. 缺乏对生成音频可控性（如特定乐器音色）的讨论
6. JavisBench 虽含 10K 样本但仍需扩展到更多语言和文化场景

## 相关工作与启发

- **MM-Diffusion (Ruan et al.)**：首个端到端 JAVG 模型，使用简单参数共享对齐
- **SyncFlow (Liu et al.)**：使用 STDiT3 块但缺乏双向信息交换
- **Seeing-Hearing (Xing et al.)**：简单嵌入对齐，缺乏细粒度空间信息
- **OpenSora (Zheng et al.)**：视频分支的预训练来源，提供了动态时间 masking 技术
- 启发：音视频同步本质上是一个条件一致性问题，层级化先验估计 + 对比学习的范式可推广到其他多模态对齐场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — HiST-Sypo 的细粒度时空先验估计具有创新性
- 实验充分度: ⭐⭐⭐⭐ — 新基准 + 新指标 + 多方法对比，但部分基线未开源
- 写作质量: ⭐⭐⭐⭐ — 结构完整，图示清晰，但部分细节在附录中
- 价值: ⭐⭐⭐⭐ — JAVG 是重要但尚未成熟的方向，本文推进了该领域的标准化
