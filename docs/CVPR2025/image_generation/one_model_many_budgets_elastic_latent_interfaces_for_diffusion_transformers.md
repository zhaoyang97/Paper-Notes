# One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers

**会议**: CVPR 2025  
**arXiv**: [2603.12245](https://arxiv.org/abs/2603.12245)  
**代码**: https://snap-research.github.io/elit  
**领域**: 图像生成  
**关键词**: DiT, 弹性推理, latent token, 计算分配, 变长接口

## 一句话总结

揭示 DiT 的计算在空间 token 上均匀分配（不会把多余计算重分配到困难区域），提出 ELIT——在 DiT 中插入可变长度的 latent interface（Read/Write 交叉注意力），训练时随机丢弃尾部 latent 学出重要性排序，推理时通过调节 latent 数量实现平滑的质量-FLOPs 权衡，ImageNet 512px 上 FID 降低 53%。

## 研究背景与动机

1. **领域现状**：DiT 通过简单的 Transformer 设计在图像/视频生成上取得 SOTA 质量，但 FLOPs 与分辨率绑定——固定计算、均匀分配到所有空间位置。
2. **现有痛点**：两个问题——(a) 推理计算预算无法灵活调整（不支持 latency-quality 权衡），(b) 简单区域和复杂区域获得相同计算量，浪费资源。
3. **核心矛盾**：用零填充实验证实——给 DiT 额外 token（零值 patch），它不会利用多余计算改善图像质量。注意力图显示零值 token 只关注其他零值 token，说明 DiT 无法重分配计算。
4. **本文要解决什么？** 让 DiT 能灵活分配计算：把更多计算给困难区域，推理时按需调节总计算量。
5. **切入角度**：引入一个 latent domain 作为计算的"中间层"——Read 层从空间 token 选择性拉取信息到 latent，核心 Transformer 块在 latent 上操作，Write 层再写回空间。
6. **核心 idea 一句话**：用可变长度的 latent interface 解耦图像分辨率与计算量，同时实现非均匀计算分配和弹性推理预算。

## 方法详解

### 整体框架

在标准 DiT 中插入三段结构：Spatial Head（几层 spatial transformer 做初步处理）→ Read（cross-attention 从 spatial 到 latent）→ Core Blocks（在 K 个 latent token 上的标准 transformer）→ Write（cross-attention 从 latent 回到 spatial）→ Spatial Tail。训练目标不变（rectified flow）。

### 关键设计

1. **Latent Interface（可变长度 latent token）**：
   - K 个可学习 latent token，通过 Read cross-attention 从空间 token 中拉取信息
   - Read 层自然学会优先关注困难区域（高损失区域），忽略简单区域（如零填充区域）
   - Write 层将更新后的 latent 广播回空间 token
   - Latent 数量是用户可控的"旋钮"——直接设定每步计算预算

2. **尾部 Token 丢弃训练**：
   - 训练时随机丢弃尾部 latent token（类似 tail dropping）
   - 效果：latent 自动学出重要性排序——前面的 latent 捕获全局结构，后面的 latent 细化细节
   - 推理时删减尾部 latent 就能平滑降低计算量，不需要重训

3. **分组机制**：
   - 空间 token 和 latent token 被分组，cross-attention 只在组内进行
   - 减少 cross-attention 复杂度，同时保持空间局部性

4. **自动引导（Autoguidance）**：
   - 用少量 latent token 的"弱模型"代替无条件模型做 CFG → 推理成本降低约 33%
   - CCFG（Cheap CFG）：在引导项中只丢类别条件，用少 token 版本做引导

### 训练策略

- Rectified Flow 目标不变
- 随机 latent 数量（multi-budget训练）：60 种预算 @512px，16 种 @256px
- 训练 500K 步，Adam，EMA β=0.9999

## 实验关键数据

### 主实验（ImageNet-1K 512px, CFG）

| 模型 | FID ↓ | FDD ↓ | IS ↑ |
|------|------|------|------|
| DiT-XL | 9.5 | 233.6 | 86.4 |
| **ELIT-DiT-XL (multi-budget)** | **4.5 (-53%)** | **98.2 (-58%)** | **147.0 (+70%)** |
| U-ViT-XL | 5.3 | 125.9 | 117.2 |
| **ELIT-U-ViT (multi-budget)** | **3.8 (-28%)** | **83.1 (-34%)** | **159.3 (+36%)** |
| HDiT-XL | 6.3 | 150.7 | 107.3 |

### 消融实验

| 配置 | 说明 |
|------|------|
| ELIT 单预算 vs 多预算 | 多预算训练始终更好（40% vs DiT 的FID降低） |
| 调节 latent 数量 vs 减少采样步数 | ELIT 的 latent 数量调节给出更优的质量-FLOPs 权衡曲线 |
| 零填充实验 | DiT 无法利用额外计算；ELIT 成功利用（匹配 DiT-B/1 的 FID） |

### 关键发现

- **ELIT 在所有测试架构上都有一致提升**：DiT (+53%)、U-ViT (+28%)、HDiT (+23%)——不是特定架构 trick 而是通用改进
- **高分辨率优势更明显**：512px 的提升远大于 256px，说明分辨率越高、像素冗余越大、动态计算重分配越有价值
- **Autoguidance 免费获得**：用少 token 版本做引导，节省 33% 推理成本且不损质量
- **与 TeaCache 兼容**：可以与 training-free 加速方法叠加使用
- **模型越大越受益**：跨 DiT-S 到 DiT-XL 的 scaling 实验显示，ELIT 的相对增益随模型增大而增加

## 亮点与洞察

- **零填充实验是极其简洁有力的 motivation**：3 行实验就揭示了 DiT 的计算均匀分配问题。这种"设计实验来暴露设计缺陷"的方法论值得学习。
- **"Drop-in"设计哲学**：ELIT 只加了两层 cross-attention，保持 RF 目标和 DiT 结构不变。这种最小化改动的设计理念使得方法真正可用——可以直接应用到 DiT/U-ViT/HDiT/MMDiT。
- **尾部丢弃 → 重要性排序**：训练时的随机丢弃让模型自动学会前面 latent 重要、后面次要。这和 NLP 的 token 重要性排序有相通之处。

## 局限性 / 可改进方向

- **Read/Write 层引入额外开销**：虽然是轻量 cross-attention，但在极短序列（低分辨率）时开销比例不可忽视
- **未测试文本到图像**：所有实验基于 ImageNet 条件生成，未在 T2I 大规模模型上验证
- **分组策略固定**：当前使用固定分组，自适应分组可能更优
- **与其他效率方法的组合空间**：与 distillation、量化等的组合未探索

## 相关工作与启发

- **vs RINs/FITs**：RIN 使用类似的 latent token + read/write，但偏离 DiT 架构且需要特殊优化器（LAMB）。ELIT 的核心优势是完全兼容标准 DiT 训练。
- **vs MaskDiT / TREAD**：这些方法在训练时丢弃 token 加速，但推理时必须恢复全部 token。ELIT 在推理时也能灵活调节。
- **vs TiTok / DCAE**：这些在 autoencoder 中使用变长 token，ELIT 把变长 token 引入生成模型内部。

## 评分

- 新颖性: ⭐⭐⭐⭐ 想法清晰优雅（latent interface + tail dropping），但 latent token 的 idea 不是全新的
- 实验充分度: ⭐⭐⭐⭐⭐ 4种架构、2种分辨率、视频生成、多种 FLOPs 曲线、zero-padding 实验
- 写作质量: ⭐⭐⭐⭐⭐ 动机实验极具说服力，图表丰富清晰
- 价值: ⭐⭐⭐⭐⭐ 对 DiT 系列的实用改进，drop-in 可直接应用到工业模型
