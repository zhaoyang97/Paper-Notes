<!-- PGen survey -->
# Personalized Generation In Large Model Era: A Survey

**会议**: ACL 2025 (Findings)
**arXiv**: [2503.02614](https://arxiv.org/abs/2503.02614)
**代码**: 无
**领域**: LLM/NLP
**关键词**: personalized generation, large language models, diffusion models, user modeling, survey

## 一句话总结
首篇跨模态个性化生成（PGen）综合综述，提出统一的用户中心视角将 NLP/CV/IR 社区的研究纳入一个框架，系统化梳理了文本/图像/视频/音频/3D/跨模态六大模态下的个性化技术、数据集和评估指标，涵盖 200+ 篇文献，并指出可扩展性、偏好演化、隐私公平等关键挑战。

## 研究背景与动机

1. **领域现状**：大模型时代，内容生成正从通用生成转向个性化生成——根据个人偏好和需求定制内容。PGen 在电商（个性化商品图）、营销（定制广告）、AI 助手等领域有重大应用价值。
2. **现有痛点**：PGen 研究分散在 NLP（文本个性化）、CV（图像/视频定制）、IR（个性化推荐）三个社区，缺乏统一跨模态视角。现有综述聚焦特定模型或任务，没有全景式概览。
3. **核心矛盾**：不同模态的数据结构和技术路线差异大，难以建立统一框架。
4. **本文要解决什么**：提供首个跨模态 PGen 综述，建立统一框架、多层分类体系，促进跨社区知识共享。
5. **切入角度**：从用户中心视角出发——PGen 本质上是"基于个性化上下文和多模态指令进行用户建模，提取个性化信号指导内容生成"。
6. **核心贡献**：统一形式化 + 多层分类法 + 应用展望 + 开放问题总结。

## 方法详解

### 统一框架

**个性化上下文**（5 维度）：用户画像（年龄/性别/职业）、用户文档（评论/邮件/帖子）、用户行为（搜索/点击/购买）、个人面部/体态、个性化主题（宠物/物品）。三个核心目标：高质量、指令对齐、个性化。

### 工作流

**用户建模**三种技术：表示学习（编码为嵌入）、Prompt Engineering（任务特定prompt）、RAG（检索增强）。

**生成建模**三步流程：
1. **基础模型选择**：LLM / MLLM / DM 根据目标模态选择
2. **引导机制**：指令引导（ICL/指令微调）+ 结构引导（adapter/cross-attention）
3. **优化策略**：Tuning-free（模型融合/多轮交互）、SFT（全量/PEFT）、偏好优化（RLHF/DPO）

### 多层分类法

| 模态 | 主要任务 | 代表方法 |
|------|---------|---------|
| **文本** | 推荐、写作助手、对话、角色扮演 | LLM-Rec, PEARL, BoB, CharacterLLM |
| **图像** | 主题驱动 T2I、人脸生成、虚拟试衣 | DreamBooth, PhotoMaker, IDM-VTON |
| **视频** | 主题驱动 T2V、说话头、姿态引导 | AnimateDiff, EMO, AnimateAnyone |
| **3D** | Image-to-3D、3D人体、3D试衣 | MVDream, DreamWaltz, DreamVTON |
| **音频** | 面部→语音、音乐生成、T2A | VoiceMe, UMP, DiffAVA |
| **跨模态** | 字幕生成、跨模态对话 | MyVLM, Yo'LLaVA |

## 实验关键数据

### 文本模态关键数据集

| 任务 | 数据集 | 个性化上下文 |
|------|--------|------------|
| 推荐 | Amazon, MovieLens, MIND | 用户行为 |
| 写作助手 | LaMP, LongLaMP, PLAB | 用户文档 |
| 对话系统 | LiveChat, FoCus | 用户画像 |
| 角色扮演 | RoleBench, HPD | 用户画像 |

### 评估维度总结

| 目标 | 文本指标 | 图像指标 |
|------|---------|---------|
| 高质量 | Perplexity, FID | FID, IS |
| 指令对齐 | BLEU, ROUGE, BERTScore | CLIP-T |
| 个性化 | 用户偏好准确率 | CLIP-I, DINO, Face Similarity |

### 关键发现与洞察
- **文本 PGen 最成熟**：LLM + RAG/ICL 组合已有成熟方案
- **图像 PGen 发展最快**：DreamBooth → Textual Inversion → InstantID 快速迭代
- **视频/3D PGen 仍在早期**：一致性保持（temporal/3D consistency）是主要瓶颈
- **跨模态 PGen 几乎空白**：同时个性化多模态输出研究极少
- **评估普遍不足**：几乎所有模态都缺乏专门衡量"个性化程度"的指标

## 亮点与洞察
- **首个跨模态统一框架**：将三个社区的 PGen 研究纳入"用户建模→生成建模"流程
- **个性化上下文 5 维度分类**：profile/document/behavior/face-body/subject 简洁清晰
- **"Deliberative Reasoning for PGen"方向**：对低频高价值场景（如广告），先深度推理再生成的思路有前景
- **Filter Bubble 警示**：PGen 可能加剧信息茧房，需要多样性增强和用户可控推理

## 局限性 / 可改进方向
- **覆盖面广但深度有限**：200+篇概览，每个方法只能简述
- **缺乏实验对比**：纯综述无实验，无定量比较
- **个性化评估指标仍是核心空白**：文中指出但未给出解决方案
- **隐私讨论偏浅**：PGen 天然需要大量用户数据，GDPR 合规只简略提及

## 相关工作与启发
- **vs LaMP (Salemi et al., 2024)**：文本个性化标准 benchmark，本综述涵盖但不限于文本
- **vs DreamBooth (Ruiz et al., 2023)**：图像 PGen 开创性工作，本综述置于更大框架下
- **vs 传统推荐系统**：推荐是"选择现有内容"，PGen 是"创造新内容"——质的飞跃

## 评分
- 新颖性: ⭐⭐⭐⭐ 首篇跨模态 PGen 综述，统一框架有价值
- 实验充分度: ⭐⭐ 纯综述无实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分类体系系统化
- 价值: ⭐⭐⭐⭐⭐ 作为跨社区资源索引极有价值
