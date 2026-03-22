# Soft Prompt Generation for Domain Generalization

**会议**: ECCV 2024  
**arXiv**: [2404.19286](https://arxiv.org/abs/2404.19286)  
**代码**: https://github.com/renytek13/Soft-Prompt-Generation-with-CGAN (有)  
**领域**: LLM/NLP  
**关键词**: domain generalization, prompt learning, CLIP, CGAN, generative model

## 一句话总结

提出 SPG（Soft Prompt Generation），首次将生成模型引入 VLM 的 prompt learning，通过 CGAN 从图像动态生成实例特定的软提示，将域知识存储在生成模型中而非提示向量中，实现更好的领域泛化性能。

## 研究背景与动机

- CLIP 等 VLM 在下游任务上通过 soft prompt 适配效果显著
- 但分布偏移（domain shift）下泛化性能大幅下降
- 现有 prompt learning 方法的不足：
  - CoOp：学习固定 prompt，过拟合训练分布
  - CoCoOp/DPL：用 MLP 生成残差向量调整固定 prompt，但简单 MLP 难以捕捉复杂的图像-提示关系
  - CAE：引入 domain bank 但提示缺乏多样性
- 核心思路转变：不再将域知识存储在 prompt 中，而是存储在生成模型中，让生成模型为每个图像动态产生域适应的 prompt

## 方法详解

### 整体框架

SPG 包含两阶段训练和一个推理阶段：
1. **阶段一**：为每个源域学习域提示标签（domain prompt labels）
2. **阶段二**：用 CGAN 学习从图像生成对应域的提示
3. **推理**：CGAN 的生成器直接为目标域图像生成实例特定的 soft prompt

### 关键设计

**1. 域提示标签学习（Training Stage I）**

- 对每个源域 d_i 单独训练一个最优 soft prompt v^{d_i}
- 使用 CoOp 框架，context length = 4
- 通过交叉熵损失优化：v^{d_i}* = argmin E[-log p(y|x, v^{d_i})]
- 域提示标签封装了丰富的域信息，作为 Stage II 的训练目标

**2. CGAN 预训练（Training Stage II）**

- 生成器 G：接收噪声 z 和图像嵌入 f(x) → 生成 soft prompt
  - 输入：[z, f(x)]（拼接噪声和 CLIP 图像特征）
  - 输出：与域提示标签形状相同的 prompt 向量
- 判别器 D：接收域提示标签/生成 prompt + 图像嵌入 → 判真假
- 对抗训练目标：min_G max_D V(G, D)
- 为增强稳定性，加入梯度裁剪策略

**3. 推理**

- 仅使用 CGAN 的生成器
- 给定目标域图像 x → f(x) + 噪声 z → G(z|f(x)) → 实例特定 soft prompt
- p(y=i|x) = softmax(⟨w_i, f(x)⟩/τ)
- 其中 w_i = g([G(z|f(x)), c_i])，g 为文本编码器

### 损失函数 / 训练策略

- Stage I：SGD 优化器，batch size 32，context length 4
- Stage II：AdamW 优化器，weight decay 1e-4
  - Learning rate: 2e-3（PACS/VLCS/TerraInc）/ 2e-4（OfficeHome/DomainNet）
  - 梯度裁剪稳定 CGAN 训练
- Backbone：ResNet50 和 ViT-B/16
- 模型选择：训练域验证集上的最高准确率

## 实验关键数据

### 主实验（Multi-source DG, ViT-B/16）

| 方法 | PACS | VLCS | OfficeHome | TerraInc | DomainNet | 平均 |
|------|------|------|-----------|----------|-----------|------|
| ZS-CLIP | 95.7 | 82.6 | 80.4 | 28.0 | 57.6 | 68.9 |
| CoOp | 95.4 | 82.5 | 82.0 | 33.0 | 56.2 | 69.8 |
| CoCoOp | 96.0 | 81.7 | 81.1 | 33.8 | 56.9 | 69.9 |
| MaPLe | 96.3 | 82.7 | 82.6 | 34.5 | 57.7 | 70.8 |
| **SPG** | **96.8** | **83.1** | **83.0** | **37.8** | **58.7** | **71.9** |

### 消融实验

| 变体 | PACS | VLCS | 平均 |
|------|------|------|------|
| w/o 域提示标签（用统一 prompt） | 95.2 | 82.0 | 降低 |
| w/o CGAN（直接用域 prompt） | 95.8 | 82.4 | 降低 |
| 用 MLP 替代 CGAN | 96.0 | 82.5 | 降低 |
| 完整 SPG | **96.8** | **83.1** | **最优** |

### 关键发现

- SPG 在 5 个 DG 基准上均取得最优，平均提升 1.1%（相对 MaPLe）
- 域提示标签是关键——为 CGAN 提供了高质量的训练目标
- CGAN 优于 MLP：能建模更复杂的图像-提示映射关系
- 在 TerraIncognita 上提升最大（+3.3% 相对 MaPLe），该数据集域差异最大
- 在 Single-source DG 和 Multi-target DG 上同样有效

## 亮点与洞察

1. **范式创新**：首次将生成模型引入 VLM prompt learning，开创"提示生成"新范式
2. **域知识存储位置的转移**：从 prompt 向量转移到生成模型参数中，更加灵活
3. **实例特定提示的多样性**：CGAN 天然支持多样化的提示生成（噪声输入带来随机性）
4. 两阶段训练策略巧妙：域提示标签作为"教师"指导 CGAN 学习域知识
5. 方法简洁易实现：基于 CGAN 这种经典模型即可取得 SOTA

## 局限性 / 可改进方向

- CGAN 训练不稳定性需要梯度裁剪等技巧
- 仅使用 CGAN 作为生成模型，更强的生成模型（扩散模型等）可能效果更好
- 域提示标签的质量直接影响 CGAN 训练，Stage I 的优化至关重要
- 未探索视觉端的 prompt 生成（仅在文本端）

## 相关工作与启发

- **CoOp**: 固定 soft prompt 的开创工作
- **CoCoOp**: 图像条件残差 prompt 的先驱
- **CGAN**: 生成模型的骨干
- **DAPL**: 域自适应中 prompt learning 的参考
- 启发：将域知识从"存储在参数中"转变为"动态生成"是提高泛化能力的有效思路

## 评分

| 维度 | 分数 (1-10) |
|------|-----------|
| 新颖性 | 8 |
| 技术深度 | 7 |
| 实验充分性 | 8 |
| 实用价值 | 8 |
| 写作质量 | 7 |
| 总体评分 | 7.6 |
