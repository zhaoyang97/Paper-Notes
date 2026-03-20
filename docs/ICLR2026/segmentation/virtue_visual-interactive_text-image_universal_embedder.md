# VIRTUE: Visual-Interactive Text-Image Universal Embedder

**会议**: ICLR 2026  
**arXiv**: [2510.00523](https://arxiv.org/abs/2510.00523)  
**代码**: [GitHub](https://github.com/sony/virtue)  
**领域**: multimodal embedding / segmentation  
**关键词**: visual prompt, embedding model, SAM2, VLM, visual-interactive, retrieval, segmentation  

## 一句话总结
提出 VIRTUE，将分割模型(SAM2)与 VLM 结合构建视觉交互式通用嵌入器，支持用户通过点/框/掩码指定兴趣区域产生实体级+全局级联合嵌入，并构建百万级 SCaR 基准评估视觉交互检索能力，在 36 个 MMEB 任务和 5 个 SCaR 任务上均达到 SOTA。

## 背景与动机
1. 现有 VLM 嵌入模型仅支持文本指令交互，缺乏视觉交互能力(点/框/掩码)
2. 视觉提示在生成模型中已广泛使用，但嵌入模型尚未探索
3. 裁剪 ROI 会丢失全局场景上下文（如"桌上的沙拉叉"裁剪后失去"桌"信息）
4. 同一图像中不同实体需要不同嵌入，但整体表示无法区分
5. 缺乏评估视觉交互嵌入能力的基准
6. 生成模型的视觉提示范式启发了嵌入模型的类似需求

## 方法详解
**架构**: SAM2 分割模型 + VLM (Qwen2-VL) + 分割-语言连接器

**三路嵌入**:
- **分割嵌入 $H_s$**: SAM2 的 prompt encoder 处理视觉提示 → mask decoder 生成 64×64 特征图 → Conv2D 压缩 → MLP 投影到 LLM 维度
- **视觉嵌入 $H_v$**: VLM 的 vision encoder 提取全局上下文
- **文本嵌入 $H_t$**: LLM 的文本嵌入层

**无视觉提示时**: 均匀采样 $N$ 个点作为替代输入 SAM2，提取多实体级特征

**训练**: 拼接 $[H_s, H_v, H_t]$ 送入 LLM，取最后 token 的 hidden state 做对比学习(InfoNCE)
- SAM2 和 vision encoder 冻结，仅训练 LoRA + 分割-语言连接器

**SCaR Benchmark**:
- 1M 样本的视觉交互图-文检索基准
- 来自 RefCOCO+/RefCOCOg/VisualGenome/COCO-Stuff/ADE20K
- 每个样本: 图像 + 框 → 检索描述实体+场景的标题
- 负样本由 GPT-4V 通过元素替换生成（对象/关系/场景三种替换策略）

## 实验关键数据
**MMEB Overall (36 tasks)**:
| 模型 | IND | OOD | Overall |
|------|-----|-----|---------|
| VLM2Vec-7B | 71.4 | 58.1 | 65.5 |
| UniME-7B | 68.4 | 57.9 | 66.6 |
| **VIRTUE-7B** | **74.4** | **61.4** | **68.6 (+2.0)** |
| **VIRTUE-2B** | **69.7** | **58.8** | **64.8 (+5.1 vs VLM2Vec-2B)** |

**SCaR (5 visual-interactive tasks)**: +15.2%–20.3% 提升
- 即使在 MMEB 通用任务上也有显著提升(3.1%–8.5%)
- 分割嵌入在非交互场景下(均匀采样点)也提供实体级信息增益

## 亮点
- **新交互范式**: 首次将视觉提示(点/框/掩码)引入嵌入模型
- **SAM2 作为结构化先验**: 比裁剪更精确地捕捉实体语义
- **兼顾通用性**: 无视觉提示时自动采样点，在传统任务上也有提升
- **SCaR 基准**: 百万级数据 + GPT-4V 生成高质量干扰项 + 多阶段过滤

## 局限性
- SAM2 增加推理计算开销（额外的分割前向传播）
- 分割-语言连接器需要从头训练，增加了训练复杂度
- SCaR 仅评估 I2T 检索，未覆盖 I2I 视觉交互场景
- 均匀采样点的自动策略可能不是最优的实体发现方式

## 相关工作
- **VLM2Vec/GME/LamRA**: VLM 嵌入模型基线，仅支持文本交互
- **CLIP/SigLIP/OpenCLIP**: 双塔嵌入模型，全局匹配
- **SAM2**: 作为实体级特征提取器
- **MMEB**: 多模态嵌入评估基准

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (视觉交互嵌入 = 全新问题定义)
- 实验充分度: ⭐⭐⭐⭐⭐ (36+5 任务 + 大量消融 + 新基准)
- 写作质量: ⭐⭐⭐⭐ (清晰系统)
- 价值: ⭐⭐⭐⭐⭐ (开辟视觉交互嵌入新方向 + 高质量基准)
