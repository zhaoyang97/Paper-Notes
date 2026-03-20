# Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse

**会议**: ACL 2025  
**arXiv**: [2412.17533](https://arxiv.org/abs/2412.17533)  
**代码**: [GitHub](https://github.com/ZILiAT-NASK/ForePLay)  
**领域**: nlp_understanding  
**关键词**: content moderation, erotic content detection, Polish NLP, annotation, dataset  

## 一句话总结
构建了首个波兰语色情内容检测数据集 forePLay（24,768 句，5 类标签），提出涵盖模糊性、暴力和社会不可接受行为的多维标注体系，评估发现专用波兰语模型显著优于多语言模型，且 Transformer 编码器模型在不平衡类别处理上表现最强。

## 研究背景与动机
1. **领域现状**：在线内容审核需求激增，但现有工具主要面向英语，对形态复杂语言（如波兰语）效果有限。
2. **现有痛点**：(a) 色情内容检测数据集稀缺且多为英语；(b) 现有数据集多用简单二分类，无法捕捉色情/暴力/社会不可接受行为等细粒度类别；(c) LLM 的安全过滤器对非英语色情内容效果不佳。
3. **核心矛盾**：波兰语等形态复杂语言中色情内容的表达方式多样，简单的二分类方案和英语中心工具难以有效检测。
4. **本文要解决什么？** 为波兰语色情内容自动检测提供高质量标注数据集和模型基准。
5. **切入角度**：从在线小说库和波兰文学作品中采样，设计 5 类排他性标注：erotic/ambiguous/violence/unacceptable/neutral。
6. **核心 idea 一句话**：用多维标注的波兰语色情内容数据集揭示语言特定模型在内容审核中的优势。

## 方法详解

### 整体框架
数据来源：69% 在线非专业小说 + 31% 波兰文学作品（含 LGBTQ+ 叙事），共 905 个文本单元。标注：6 名标注者（3 男 3 女），每句 3 人独立标注 + 多数投票 + 超级标注者仲裁。模型评估：编码器模型（HerBERT、RoBERTa）微调 + 波兰 LLM（PLLuM 系列、Bielik）+ 通用 LLM（GPT-4o、Llama-3.1 等）零/少样本。

### 关键设计
1. **五类标注体系**:
   - erotic (25.68%): 性活动描述、明显色情暗示
   - ambiguous (5.43%): 可能在特定语境下引发色情联想但本身中性
   - violence (0.28%): 性骚扰/强奸/非自愿暴力
   - unacceptable (0.47%): 恋童癖/恋兽癖/乱伦等非法/禁忌行为
   - neutral (68.14%): 其余内容
   - 层级规则：unacceptable > violence > erotic > ambiguous > neutral

2. **标注质量控制**:
   - Cohen's Kappa 大多在 0.66-0.72（排除异常标注者 Fem1 后 Krippendorff's Alpha = 0.716）
   - 每作者最多 2 篇故事以避免过拟合特定写作风格

### 四种数据集配置
- Basic：二分类（neutral/erotic）
- Core：三分类（+ambiguous）
- Extended：四分类（+合并 violence+unacceptable）
- Full：完整五分类

## 实验关键数据

### 编码器模型（微调）
| 模型 | Basic F1 | Core F1 | Full F1 |
|------|----------|---------|---------|
| HerBERT-Large | 0.939 | 0.738 | 0.648 |
| RoBERTa-Base | 0.944 | 0.738 | 0.707 |
| RoBERTa-Large | 0.943 | 0.748 | 0.664 |

### LLM 零样本对比
| 模型 | Basic F1 | Core F1 | Full F1 |
|------|----------|---------|---------|
| GPT-4o (0-shot) | 0.888 | 0.640 | 0.340 |
| PLLuM-Mistral-12B | 0.894 | 0.656 | 0.401 |
| PLLuM-Mixtral-8x7B | 0.874 | 0.647 | - |
| Bielik-11B (5-shot) | 0.868 | 0.607 | 0.480 |

### 关键发现
- 波兰语专用编码器模型一致优于通用多语言 LLM（RoBERTa-Base Full F1=0.707 vs GPT-4o=0.340）
- 随分类粒度增加（Basic→Full），所有模型性能显著下降
- ambiguous 类别标注一致性最低，是检测难点
- 波兰语 LLM（PLLuM）在零样本下优于相同架构的通用 LLM

## 亮点与洞察
- 多维标注体系（特别是 ambiguous 和 violence/unacceptable 的区分）比简单二分类更贴近真实内容审核需求
- 揭示了微调小模型在特定语言内容审核中可大幅超越大规模通用 LLM 的现象
- 数据集包含 LGBTQ+ 内容的有意覆盖，避免了这类叙事被系统性忽略

## 局限性 / 可改进方向
- 暴力和社会不可接受类别样本极少（<1%），模型难以学习
- 仅句子级标注，缺乏文档级上下文
- 仅波兰语，方法论可迁移但数据不直接适用于其他语言
- 标注者间变异（HLV）在 ambiguous 类上仍较高

## 相关工作与启发
- **vs Jigsaw/BeaverTails**: 英语中心+二分类，forePLay 提供波兰语+五分类
- **vs CENSORCHAT**: 面向对话系统监控，forePLay 面向文本内容检测
- **vs Llama Guard**: 通用安全分类器，本文证明语言专用模型在非英语场景更优

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个波兰语色情内容检测数据集，标注体系有特色
- 实验充分度: ⭐⭐⭐⭐ 编码器+LLM+开源/闭源全面对比
- 写作质量: ⭐⭐⭐⭐ 标注流程描述详细，伦理考量到位
- 价值: ⭐⭐⭐ 对波兰语 NLP 社区直接有用，跨语言方法可迁移性一般
