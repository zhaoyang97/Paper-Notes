# SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models

**会议**: ICLR2026  
**arXiv**: [2506.01062](https://arxiv.org/abs/2506.01062)  
**代码**: [HuggingFace](https://huggingface.co/datasets/vtllms/sealqa)  
**领域**: llm_reasoning  
**关键词**: benchmark, search-augmented LLM, RAG, noisy retrieval, test-time scaling  

## 一句话总结
提出SealQA挑战基准，包含111道使前沿非推理模型准确率为0的事实性问题，专门评估搜索增强LLM在噪声/冲突/误导性检索结果下的推理能力。

## 背景与动机
1. 现有RAG/搜索增强基准聚焦简单事实查找，top-ranked结果即可直接回答，无法反映真实搜索的混乱本质
2. 现实搜索常返回过时、误导或表面相关但无用的文档，需深层推理来筛选不一致信息
3. LLM已在MMLU等基准上达>90%，传统基准趋于饱和
4. Test-time scaling（o系列/R1等推理模型）是否真能可靠提升复杂搜索推理？需系统验证
5. 需要小而精的高质量挑战基准（类似GPQA-Diamond路线）

## 方法
**数据构建**: NLP研究者团队(6人8个月)手工构建问题，每题平均花费>1小时。要求搜索结果触发歧义/冲突/噪声。多轮审查(2+研究生→专家审批)。

**三种变体**:
- **Seal-0** (111题): 使GPT-4o/4.1等前沿模型在10-15次尝试中准确率为0的核心集
- **Seal-Hard** (254题): 包含Seal-0+其他高难度题
- **LongSeal** (254题): 每题配1个gold文档+最多50个hard negative，测试长上下文多文档推理

**问题类型**: Q1高级推理(72.4%), Q2实体消歧(58.3%), Q3时间追踪(13.7%), Q4跨语言推理(5.5%), Q5假前提检测(4.3%)

**评估**: GPT-4o-mini自动评分器，与人类判断98%一致

## 实验
| 模型 | Seal-0 (w/search) | Seal-Hard (w/search) |
|------|:---:|:---:|
| GPT-5-high | **43.2** | **63.8** |
| GPT-5-mini-high | 41.4 | 60.2 |
| o3-high | 14.4 | 32.7 |
| DeepSeek-R1-671B | 1.8 | 11.0 |
| GPT-4.1 | 0.0 | 20.5 |
| GPT-4o | 0.0 | 15.0 |
| 人类(研究生+Google) | — | ~62.0 |

**关键发现**: (1) Seal-0上即使GPT-5+工具也仅43.2%，说明极具挑战性; (2) DeepSeek-R1加搜索后反而从22.4%降至11.0%(Seal-Hard)，推理模型对噪声检索脆弱; (3) 增加test-time compute不能可靠提升性能，常出现平台期或下降; (4) 跨语言推理(Q4)和假前提检测(Q5)是最薄弱环节; (5) LongSeal中模型虽缓解了"lost-in-the-middle"问题，但仍无法可靠识别gold文档。

## 亮点
- 极端对抗性构建：使前沿模型"归零"的benchmark设计理念令人印象深刻
- 揭示推理模型+搜索的反直觉失败模式（噪声搜索严重干扰推理）
- Test-time scaling无效发现具有重要启示意义
- 动态版本化benchmark，承诺定期更新答案

## 局限
- 样本量小(111/254题)，统计功效有限
- 仅英文问题（虽含Q4跨语言检索需求）
- GPT-5等模型可能因数据污染受益，难以完全控制
- 人工构建成本极高，可扩展性差
- 问题类型分布不均匀(Q4/Q5占比极少)

## 相关工作
- 搜索增强评估: FreshQA (Vu et al. 2024) 关注时效性; SimpleQA (Wei et al. 2024) 对抗GPT-4构建
- 挑战基准: GPQA-Diamond (Rein et al. 2024) 198题专家级QA; MMLU-Pro扩展版
- 长上下文: Lost-in-the-middle (Liu et al. 2024); needle-in-a-haystack测试

## 评分
- 新颖性: ⭐⭐⭐⭐ (对抗性搜索场景构建新颖)
- 实验充分度: ⭐⭐⭐⭐⭐ (覆盖大量开闭源模型+人类基线)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图表丰富)
- 价值: ⭐⭐⭐⭐ (揭示RAG+推理的关键弱点)
