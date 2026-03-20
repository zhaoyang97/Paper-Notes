# CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter

**会议**: ACL 2025  
**arXiv**: [2502.16880](https://arxiv.org/abs/2502.16880)  
**代码**: 无  
**领域**: 模型压缩 / LLM效率  
**关键词**: speculative decoding, draft model, representation alignment, LM head compression, vocabulary  

## 一句话总结
CORAL 通过跨步表示对齐（CSRA）改进多步训练中 draft 模型的特征一致性，并用权重分组机制压缩大词表 LM head 的推理延迟，在 LLaMA3/Qwen2.5 上实现 2.50-4.07× 加速，超越 EAGLE-2 和 HASS。

## 研究背景与动机

1. **领域现状**：投机解码用轻量 draft 模型预生成 token，target 模型并行验证。EAGLE 使用 target 模型的 hidden states 训练 draft 模型，但存在训练-推理不对齐（训练用 target states，推理用自身 states）。HASS 引入多步训练让 draft 模型适应自身特征，但多步输入差异大使训练难收敛。
2. **现有痛点**：(1) 多步训练中不同步骤的输入特征差异大，轻量 draft 模型难以适应；(2) 不同步骤间可能存在梯度冲突；(3) 现代 LLM 词表越来越大（LLaMA3 128K, Qwen2.5 152K），LM head 成为 draft 模型的延迟瓶颈——如 Qwen2.5-7B 的 draft 模型中 LM head 占总延迟的主要部分。
3. **核心矛盾**：多步训练提升了 draft 精度但引入了特征不一致问题；大词表提升了 LLM 能力但拖慢了 draft 模型速度。
4. **本文要解决什么？** 同时解决多步训练的特征一致性问题和大词表的延迟问题。
5. **切入角度**：用对比学习强制多步训练的特征保持一致 + 用路由分组选择性激活 LM head 参数。
6. **核心idea一句话**：跨步对比对齐使 draft 特征在不同训练步骤间保持一致 + LM head 参数分组激活解决大词表延迟。

## 方法详解

### 整体框架
多步训练：draft 模型在每步输出特征 → CSRA 用对比学习约束不同步骤的同位置特征保持一致 → 同时用标准 CE 损失训练分类。推理：draft 模型生成 token → LM head 路由选择性激活参数子集 → target 模型验证。

### 关键设计

1. **Cross-Step Representation Alignment (CSRA)**:
   - 做什么：在多步训练中，强制 draft 模型不同步骤对同一位置输出的特征保持一致
   - 核心思路：用对比学习——同位置不同步骤的特征作为正对，不同位置的特征作为负对。最小化正对之间的距离
   - 设计动机：多步训练中 step 1 用 target states，step 2/3 用自己的 states，特征分布可能差异大。CSRA 强制一致性使 draft 模型更稳定
   - 与 HASS 的区别：HASS 仅做多步训练但不约束特征一致性

2. **LM Head 权重分组**:
   - 做什么：将大词表 LM head 的权重按 token embedding 相似度分组，推理时只激活相关组
   - 核心观察：大词表 LM head（128K-152K 词）占 draft 模型 50%+ 延迟。如 Qwen2.5 的 draft 模型中 LM head 参数量超过 transformer 层
   - 方法：用路由器预测当前 token 属于哪个权重组 → 只激活该组参数做矩阵乘法
   - 效果：大幅降低 draft 模型延迟，特别是对大词表模型

## 实验关键数据

### 主实验：加速比（温度=0）

| 方法 | LLaMA3-8B | Qwen2.5-7B | LLaMA2-7B |
|------|----------|-----------|----------|
| Vanilla 解码 | 1.0× | 1.0× | 1.0× |
| EAGLE-2 | ~2.5× | ~2.3× | ~2.8× |
| HASS | ~2.7× | ~2.5× | ~3.0× |
| **CORAL** | **~3.2×** | **~3.5×** | **~4.07×** |

### 消融实验

| 配置 | 加速比 | 说明 |
|------|------|------|
| CORAL 完整 | 最佳 | CSRA + LM head 分组 |
| w/o CSRA | 下降明显 | 多步训练特征不一致导致精度低 |
| w/o LM head 分组 | 下降（延迟） | 大词表模型延迟增加 |
| HASS + LM head 分组 | 中等 | 说明两个贡献独立有效 |

### 关键发现
- **大词表是 draft 模型的新瓶颈**：LLaMA3(128K词)和 Qwen2.5(152K词)的 draft 模型中 LM head 延迟占比从 LLaMA2(32K词)的~20%增至~50%+
- **CSRA 显著提升接受率**：跨步对齐使 draft 模型的平均接受长度 τ 提升
- **大词表模型上 CORAL 优势更明显**：因为 LM head 压缩在大词表上收益更大

## 亮点与洞察
- **发现 LM head 是 draft 模型的新瓶颈**：随着 LLM 词表扩大到 100K+，这个被忽视的问题越来越严重。识别并解决这个问题是重要贡献
- **对比学习在投机解码训练中的应用**：CSRA 用对比学习约束多步训练的一致性是自然而有效的
- **实用加速效果突出**：2.50-4.07× 是投机解码领域的强结果

## 局限性 / 可改进方向
- LM head 分组需要额外的路由器，引入微小额外延迟
- 分组策略基于 token embedding 相似度，可能对 OOD token 不鲁棒
- CSRA 的对比学习超参需要调节
- 未在 MoE 模型上验证

## 相关工作与启发
- **vs EAGLE-2 (Li et al., 2024)**: EAGLE 用 target states 训练+tree attention。CORAL 通过 CSRA 提升多步训练精度，通过 LM head 压缩提升速度，全面超越
- **vs HASS (Zhang et al., 2024)**: HASS 提出多步训练解决训练-推理不对齐。CORAL 在此基础上加 CSRA 一致性约束和 LM head 加速
- **vs CLaSp**: CLaSp 是无训练的层跳策略（加速有限但即插即用）。CORAL 需要训练 draft 模型但加速更大

## 评分
- 新颖性: ⭐⭐⭐⭐ 发现 LM head 瓶颈+CSRA 一致性约束，两个贡献独立且有效
- 实验充分度: ⭐⭐⭐⭐ 三个 LLM 系列三个基准，消融完整
- 写作质量: ⭐⭐⭐⭐ 分析驱动的方法设计，逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 4.07× 加速的实际部署价值很高
