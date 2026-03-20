# AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network

**会议**: CVPR 2026  
**arXiv**: [2603.12659](https://arxiv.org/abs/2603.12659)  
**代码**: [https://github.com/yuhu990424/AVION](https://github.com/yuhu990424/AVION) (有)  
**领域**: 遥感 / 视觉语言模型  
**关键词**: 遥感VLM适配、知识蒸馏、Prompt Tuning、文本原型增强、少样本分类  

## 一句话总结
提出 AVION 蒸馏框架，通过 LLM 生成并视觉验证的文本原型解决遥感 VLM 的"语义贫乏"，通过双模态 Prompt Tuning 解决"视觉刚性"，在 6 个遥感基准上实现少样本和 base-to-novel 同时提升。

## 背景与动机
将通用 VLM 适配到遥感领域面临两个核心挑战：(1) **语义贫乏**——遥感数据集通常只提供类别名称，而同一类别外观差异极大，简单模板无法刻画细粒度差异；(2) **视觉刚性**——多数 PEFT 方法仅更新文本编码器而冻结视觉编码器，使模型难以捕捉遥感的尺度变化和俯视角特征。现有方法在 base 类上提升的同时会严重损害 novel 类泛化。

## 核心问题
如何在轻量级参数高效微调框架下，同时解决遥感 VLM 的文本语义不足和视觉特征适应不足问题，且不牺牲对未见类的泛化能力？

## 方法详解
### 整体框架
离线阶段用大教师模型构建高质量文本原型；在线阶段训练轻量级学生模型（仅更新 Prompt 参数），推理时教师不参与。

### 关键设计
1. **LLM 域提示 + RS-Flag 过滤**: 用 Gemini 2.5 Flash 为每个类生成最多 50 条遥感视角描述。通过 RS-Flag 规则过滤：必须含遥感正向词（overhead/aerial view/satellite imagery 等）、不含负向词（street/indoor/selfie 等）、长度 6-20 词。
2. **选择性原型聚合**: 用教师视觉编码器计算各类视觉原型（类内图像嵌入平均），与 LLM 描述计算余弦相似度，用 MAD-based 鲁棒 z-score 剔除离群描述，再以相似度 + RS-Flag 作为权重做 softmax 聚合得到最终文本原型。本质上是一个无参数交叉注意力机制。
3. **三方面对齐蒸馏**: 视觉对齐 L_img（学生视觉嵌入向教师视觉嵌入）、文本对齐 L_text（学生文本嵌入向教师增强原型）、相似度对齐 L_logit（KL 散度匹配师生的完整类间分布）。

### 损失函数 / 训练策略
总损失 L = L_task + 0.5*L_img + 0.5*L_text + 1.0*L_logit。L_logit 使用 30% 线性 warm-up 确保收敛稳定，蒸馏温度 tau=2。AdamW，lr=5e-4，batch 4。学生 GeoRSCLIP(ViT-B/32)，教师 GeoRSCLIP(ViT-H/14)，仅 98,304 可训练参数（<1% backbone）。单 NVIDIA L4 GPU。

## 实验关键数据
| 任务 | 指标 | AVION | 之前SOTA | 提升 |
|------|------|-------|---------|------|
| 少样本 1-shot (6数据集均) | Accuracy | 74.27% | 74.27% (APPLeNet) | 并列 |
| 少样本 8-shot | Accuracy | 92.45% | 89.79% | +2.65pp |
| 少样本 16-shot | Accuracy | 93.69% | 90.86% | +2.83pp |
| Base-to-Novel (HM) | HM | 87.05% | 84.20% (PromptKD) | +2.85pp |
| Base-to-Novel (Novel) | Novel | 79.94% | 79.75% (GeoRSCLIP ZS) | 唯一超越ZS |
| RSITMD 检索 | mR | +1.11pp | vs GeoRSCLIP-FT | - |

### 消融实验要点
- B0->B7 递增消融：深度 Prompt 提升 base 但损 novel -16.01pp；视觉对齐恢复 HM +6.03pp；文本原型增强累计 HM +10.31pp 为最大贡献；logit 对齐+warm-up 再 +4.00pp。
- 原型消融 P0-P6：手工模板 79.52% HM -> LLM+域约束+选择性聚合 83.05% HM。
- LLM 选择：GPT-5/Llama-3.1-70B/Gemini 2.5 Flash 结果差异极小（HM 86.00-87.05%），框架对 LLM 不敏感。
- ImageNet 上也有效：HM 76.94% vs MMRL 74.45%。

## 亮点
- AVION 是对比中唯一在 base 和 novel 类上同时超越 zero-shot 基线的 PEFT 方法
- 三方面对齐设计优雅：嵌入级（视觉+文本）和行为级（logit）互补
- 算力友好：可训练参数不到 backbone 的 1%
- 选择性原型聚合是一个通用技术，可扩展到其他域特定 VLM 适配

## 局限性 / 可改进方向
- 仅验证光学遥感，SAR/高光谱等模态未涉及
- RS-Flag 规则较硬编码，对新场景可能需要手动调整 token 列表
- 教师模型固定为同系列的更大模型，跨架构蒸馏效果未知

## 与相关工作的对比
- vs CoOp/CoCoOp: AVION novel 类显著更好（79.94% vs 69.52%/70.69%）
- vs APPLeNet: 冻结视觉编码器导致 novel 类文-图嵌入错位，t-SNE 清晰可见
- vs MMRL: 双侧开放但缺乏语义丰富的文本引导，novel 类多模态对齐更差
- vs PromptKD: AVION HM 87.05% vs 84.20%

## 启发与关联
- LLM 生成 + 视觉验证 的文本原型构建思路可迁移到医学/工业等其他专业域
- 三方面对齐的蒸馏损失设计值得在其他 PEFT 框架中尝试

## 评分
- 新颖性: ⭐⭐⭐⭐ 文本原型增强+选择性聚合的组合设计优雅，解决了真实痛点
- 实验充分度: ⭐⭐⭐⭐⭐ 6个分类+2个检索基准，三协议，详尽消融和敏感性分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表直观，附录完整
- 价值: ⭐⭐⭐⭐ 遥感 PEFT 方向的扎实工作，方法有通用性
