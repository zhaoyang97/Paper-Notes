# Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates

**会议**: ACL 2025  
**arXiv**: [2505.22943](https://arxiv.org/abs/2505.22943)  
**代码**: [https://vision.snu.ac.kr/projects/mac](https://vision.snu.ac.kr/projects/mac)  
**领域**: 多模态VLM  
**关键词**: 组合性漏洞, 对抗攻击, CLIP, 自训练, 多模态表征  

## 一句话总结
提出MAC基准和diversity-promoting自训练方法，通过让LLM生成欺骗性文本来系统暴露CLIP等预训练多模态表征的组合性漏洞，在图像/视频/音频三个模态上均显著超越已有方法。

## 研究背景与动机
1. **领域现状**：CLIP等预训练多模态表征已成为检索、生成、奖励建模等下游任务的核心组件，其质量直接影响整个系统。
2. **现有痛点**：这些表征存在严重的组合性漏洞——例如，CLIP可能给"a bed is sitting on a baby"比"a baby is sitting on a bed"更高的相似度分数。现有基准（Winoground、SugarCrepe等）只针对特定模态（图像）和预定义的文本操作类型（替换、交换），无法全面暴露漏洞。
3. **核心矛盾**：(a) 基于规则的方法（词交换等）生成的负样本不自然、容易防御；(b) 人工标注成本高、难以规模化；(c) 现有方法只评估攻击成功率，忽略了攻击样本集的多样性——单调的攻击模式容易防御且无法揭示多样化的漏洞。
4. **本文要解决什么？** (a) 提出模态无关的组合性漏洞评估框架；(b) 同时评估攻击成功率和多样性；(c) 用小模型（8B）实现高效的漏洞发现。
5. **切入角度**：将"LLM能否欺骗CLIP"形式化为对抗攻击问题，定义多维评估准则（跨模态、单模态、距离、辅助），并用rejection sampling自训练+diversity-promoting选择使LLM学会生成更有效且多样的对抗文本。
6. **核心idea一句话**：用LLM自训练生成欺骗性文本，通过多维攻击成功率+熵值多样性双重评估来系统基准测试多模态表征的组合性漏洞。

## 方法详解

### 整体框架
给定多模态数据对$(t_i, x_i)$（文本+图像/视频/音频），用LLM生成器$g$产生对抗文本$\tilde{t}_i$，使目标表征$f$（如CLIP）错误地认为$\tilde{t}_i$比原始$t_i$与$x_i$更匹配。然后通过sample-wise四维准则评估每个样本的攻击成功，通过group-wise熵值评估整个攻击集的多样性。

### 关键设计

1. **MAC四维评估准则（sample-wise）**:
   - 做什么：定义攻击成功的严格多维条件
   - 核心思路：攻击成功需同时满足四个条件：(i) 跨模态准则：$d_\theta(y_{t_i}, y_{x_i}) < d_\theta(y_{\tilde{t}_i}, y_{x_i})$，即欺骗模型认为对抗文本与原始模态更匹配；(ii) 单模态准则：NLI模型判定$\tilde{t}_i$与$t_i$不构成蕴含关系（不是简单改写）；(iii) 距离准则：Levenshtein编辑距离 < 平均token长度的一半（限制修改幅度）；(iv) 辅助准则：遵循预定义规则（如指定的操作类型、排除否定等捷径）。总攻击成功率 $R = \frac{1}{M_D}\sum_i (s_i^c \cdot s_i^u \cdot s_i^d \cdot s_i^a)$
   - 设计动机：任何单一准则都不足以定义有效攻击——仅看跨模态容易退化为改写，仅看编辑距离忽略语义区分。四维准则形成对攻击质量的全面约束

2. **Group-wise多样性评估**:
   - 做什么：衡量整个对抗样本集使用了多少种不同的文本变换模式
   - 核心思路：为每个样本对$(t_i, \tilde{t}_i)$构建属性增强token（OP_POS_LEMMA格式，如I_NOUN_man表示插入名词man），然后计算所有token集合的熵 $H = -\sum_j p_j \log p_j$ 和distinct-1 $D_1$。更高的$H$表示更多样化的攻击模式
   - 设计动机：如果攻击总是使用相同词汇（如总是换man/woman），虽然成功率高但容易防御，且无法揭示表征的多种漏洞

3. **Diversity-Promoting Self-Training**:
   - 做什么：训练小LLM（Llama-3.1-8B）自动生成高攻击成功率且多样化的对抗文本
   - 核心思路：三步流程。(i) 用基础LLM对每个训练样本生成N=64个候选；(ii) 通过Gibbs采样式迭代选择：为每个样本点从其成功攻击候选中选择能最大化全局熵$H$的样本（Algorithm 1），循环K轮；(iii) 用选出的多样化成功样本做RFT（rejection sampling fine-tuning），loss为标准自回归：$\mathcal{L} = -\frac{1}{M_{\hat{D}}}\sum_i\sum_j \log g(\tilde{t}_{i,j}|\tilde{t}_{i,<j}, \mathcal{I}, t_i; \Theta)$
   - 设计动机：朴素自训练只用随机选择的成功样本训练，会导致模型单调化（总生成类似模式）。通过在训练数据选择阶段引入多样性优化，让模型学到更丰富的攻击模式

### 损失函数 / 训练策略
采用RFT损失（标准自回归交叉熵）。训练数据来自N=64的大规模采样后经diversity-promoting选择的成功攻击样本。推理时只需N=4即可获得高性能。

## 实验关键数据

### 主实验
在三个模态上的攻击效果对比（N=4, 本文方法 vs 最佳baseline）：

| 模态/数据集 | 指标 | 本文 (Total ASR) | 最佳Baseline | 提升 |
|-------------|------|-----------------|-------------|------|
| Image/COCO (CLIP) | Total ASR↑ | 42.10% | 23.33% (SeeTrue) | +18.77pp |
| Video/MSRVTT (LB) | Total ASR↑ | 45.60% | 36.90% (VFC) | +8.70pp |
| Audio/AudioCaps (LB) | Total ASR↑ | 52.87% | 5.76% (CompA) | +47.11pp |

多样性对比（N=4, Image/COCO）：

| 方法 | H↑ | D1↑ |
|------|-----|------|
| 本文 (Diversity-Promoted) | **7.747** | **0.129** |
| Self-Train (无diversity) | 7.507 | 0.120 |
| Zero-shot | 7.571 | 0.130 |
| SeeTrue | 7.168 | 0.124 |

### 消融实验
自训练各组件的贡献（Image/COCO, N=4）：

| 配置 | Cross ASR | Total ASR | H |
|------|-----------|-----------|-----|
| Zero-shot | 37.29% | 19.19% | 7.571 |
| + Self-Train | 43.08% | 34.64% | 7.507 |
| + Large-N Distilled | 48.29% | 42.03% | 7.452 |
| + Diversity-Promoted (Full) | 47.93% | 42.10% | **7.747** |

### 关键发现
- **自训练显著提升攻击成功率**：从zero-shot的19.19%到self-train的34.64%（+15.45pp），说明LLM可以通过学习自己生成的成功样本大幅提升漏洞发现能力
- **跨模型迁移性良好**：在CLIP上训练的攻击对SigLIP、NegCLIP、BLIP也有效（ASR 23-29%），说明组合性漏洞在不同表征模型间共享
- **ASR与多样性的trade-off**：朴素自训练提升ASR但降低多样性（H从7.571降至7.507），diversity-promoting选择恢复并提升多样性（7.747）而几乎不损失ASR
- **小模型不输大模型**：Llama-3.1-8B的攻击效果不逊于GPT-4o（甚至在某些设置下更好），说明漏洞发现不需要昂贵的大模型
- **音频模态最脆弱**：Audio/AudioCaps上Total ASR达52.87%，远高于Image（42.10%）和Video（45.60%），说明音频-语言表征的组合性漏洞最严重

## 亮点与洞察
- **模态无关的统一评估框架**：将组合性漏洞从视觉-语言扩展到视频和音频，用统一的四维准则+熵值多样性评估，这是此前工作没有做到的
- **Diversity-promoting选择的巧妙设计**：不改变训练loss，而是在训练数据选择阶段引入diversity优化（Gibbs采样最大化全局熵），简单但有效地解决了自训练单调化问题。这个思路可以迁移到任何rejection sampling场景
- **属性增强token设计**：OP_POS_LEMMA的编码方式把文本变换结构化为可量化比较的token，使得diversity有了可计算的度量

## 局限性 / 可改进方向
- **仅修改文本不修改模态输入**：只通过文本变换攻击，未探索图像/视频/音频侧的联合攻击
- **评估依赖NLI模型**：单模态准则使用NLI模型判断蕴含关系，NLI本身的错误会影响评估
- **对防御策略的分析不足**：知道了漏洞，但如何利用这些发现来增强模型的组合性没有深入探讨
- **可改进方向**：利用发现的对抗样本做对比学习数据增强来修复CLIP的组合性漏洞

## 相关工作与启发
- **vs SugarCrepe**: SugarCrepe用ChatGPT生成负样本但只做图像且不考虑多样性，本文提供了更严格的多维评估和diversity-promoting机制
- **vs RoCOCO**: RoCOCO用规则方法在不同词汇选择策略间展现ASR-diversity trade-off，本文的自训练方法打破了这一trade-off
- **vs CompA**: CompA只做音频，本文统一了三个模态的评估，且在音频上大幅超越CompA

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将对抗攻击+多样性评估统一应用于多模态组合性漏洞基准测试
- 实验充分度: ⭐⭐⭐⭐⭐ 三模态、多目标模型、跨模型迁移、消融实验齐全
- 写作质量: ⭐⭐⭐⭐ 问题形式化清晰，评估准则定义严格
- 价值: ⭐⭐⭐⭐ 揭示CLIP等核心表征的系统漏洞，对VLM安全性研究有重要意义
