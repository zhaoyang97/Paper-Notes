# Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning

**会议**: ACL 2025  
**arXiv**: [2411.17679](https://arxiv.org/abs/2411.17679)  
**代码**: [https://github.com/FloatFrank/TIPA](https://github.com/FloatFrank/TIPA)  
**领域**: LLM NLP / 字符级理解  
**关键词**: Token Internal Structure, Character-Level Understanding, BPE Tokenization, Chinese Spelling Correction, Position Awareness  

## 一句话总结

提出 TIPA（Token Internal Position Awareness）方法，通过在 tokenizer 词汇表上进行逆序字符预测训练，增强 LLM 对 token 内部字符结构和位置的感知能力，显著提升中文拼写纠错等字符级任务的表现。

## 研究背景与动机

1. **领域现状**:
   - 主流 LLM 使用 BPE 等 subword 分词方法，将文本切分为子词单元以提高计算效率
   - LLM 的训练以 next-token prediction 为核心，关注 token 间的序列依赖关系
   - 这种设计天然隐藏了 token 内部的字符结构信息

2. **现有痛点**:
   - LLM 无法准确感知 token 内部字符的位置和组成，例如被问到 "strawberry 有几个 r" 时经常回答错误
   - 中文场景更为严重：一个 token 可能包含多个汉字，模型难以定位具体汉字在原文中的位置
   - 这严重影响了中文拼写纠错（CSC）等需要精确字符定位的任务
   - 现有的 byte-level 模型（如 ByT5）需要改变架构，无法低成本适配现有 subword LLM

3. **核心矛盾**:
   - BPE 分词提高了计算效率，但牺牲了字符级信息的可见性
   - 模型学会了 token 之间的关系，但对 token 内部的字符顺序和位置几乎"无感"

4. **本文要解决什么？**
   - 在不更改模型架构的前提下，增强 LLM 对 token 内部字符结构和位置的理解能力

5. **切入角度**:
   - 利用 tokenizer 自身的词汇表构造训练数据，通过逆序字符预测任务教会模型理解 token 内部结构

6. **核心idea一句话**:
   - 用逆序的方式让模型输出 token 内每个字符的位置和内容（如"girl" → {4:"l", 3:"r", 2:"i", 1:"g"}），从而隐式学习 token 的内部字符结构

## 方法详解

### 整体框架

TIPA 系统包含两个核心组件：
1. **TIPA**：基于 tokenizer 词汇表的单 token 逆序字符位置训练
2. **MTIPA**：扩展到多 token 句子级别的字符位置训练

### 关键设计

1. **TIPA (Token Internal Position Awareness)**:
   - 做什么：对 tokenizer 词汇表中的每个 token，生成逆序的字符-位置映射作为训练数据
   - 核心思路：给定 token t，将其拆解为字符序列 [c₁, c₂, ..., cₙ]，然后构造逆序映射 {n: cₙ, n-1: cₙ₋₁, ..., 1: c₁}
   - 训练 prompt 示例：输入 "girl"，输出 {"4": "l", "3": "r", "2": "i", "1": "g"}
   - 设计动机：逆序输出使得第一个数字即为 token 长度，巧妙地将分词、长度信息、位置信息统一到单一任务中
   - 仅使用 tokenizer 词汇表中可以 UTF-8 表示的 token，不需要外部数据

2. **MTIPA (Multi-Token Internal Position Awareness)**:
   - 做什么：将逆序字符预测扩展到完整句子层面
   - 核心思路：从目标任务的训练数据中采样句子，对整句进行字符分解和逆序位置映射
   - 采样比例 r 设为较小值（如 10%），平衡数据量和训练效率
   - 设计动机：MTIPA 专门用于需要精确字符位置预测的任务（如带位置预测的 CSC）

3. **重新定义 CSC 任务**:
   - 做什么：将传统的"输出纠正后全句"改为"输出错误字符的位置、错误字符和正确字符"
   - 核心思路：如"业内人事称..." → [{"position": 4, "incorrect": "事", "correction": "士"}]
   - 设计动机：大幅减少输出 token 数量（position 方法比传统方法少 36-51% 的 token），提高效率

4. **扩展方法：Full-Parameter SFT**:
   - 将 TIPA 数据与 tulu-3-sft-mixture 数据合并，对 Llama-3.1-8B 进行全参数微调
   - 产出 Llama-3.1-Tulu-TIPA-8B，在保持模型通用能力的同时增强字符级处理

### 损失函数 / 训练策略

- 使用标准的 SFT 训练（监督微调），支持 LoRA 和全参数微调两种方式
- TIPA 数据来自 tokenizer 词汇表，无需外部标注数据
- 训练增加的时间很少，推理时不引入任何额外延迟

## 实验关键数据

### 主实验

**实验1：带位置预测的 CSC 任务**
- 引入新评估指标 PPA（Position Prediction Accuracy）衡量模型定位错误字符的能力
- TIPA + MTIPA 显著提升了模型的字符位置预测准确率

**实验2：传统 CSC 任务**
- 在 CSCD-Test 和 Lemon 数据集上，TIPA 提升了模型的拼写纠错能力
- 即使不显式要求位置预测，TIPA 也能通过增强字符理解来改善纠错效果

**输出 Token 数对比**：
| 数据集 | 传统方法 Token 数 | 位置方法 Token 数 |
|--------|-------------------|-------------------|
| Train | 8,905,800 | 8,016,111 |
| CSCD-Test | 188,310 | 54,897 |
| Lemon | 532,684 | 258,112 |

- 位置方法输出 Token 减少了约 51-71%

### 关键发现

1. **逆序优于正序**：逆序预测隐式编码了 token 长度信息，效果优于正序
2. **TIPA 的通用性**：即使在不需要精确位置预测的下游任务中，TIPA 仍然有效
3. **LoRA vs 全参数微调**：全参数微调结合 tulu-3 数据可以在保持通用能力的同时增强字符级理解
4. **MTIPA 数据量需控制**：MTIPA 数据过长可能导致训练时间过长，且 LoRA 学习大量长度推理信息可能降低特定任务能力
5. **GPT-4o 也存在字符定位问题**：即使是最强的闭源模型，也存在 token 内字符位置感知不足的问题

## 亮点与洞察

- **利用 tokenizer 自身词汇表**：非常巧妙地利用现有资源构造训练数据，无需外部标注
- **逆序设计的精巧之处**：一个简单的逆序操作同时解决了分词、长度、位置三个子问题
- **重新定义 CSC 任务范式**：从输出全句改为输出位置+字符，大幅降低推理成本
- **揭示 BPE 的固有局限**：系统性地展示了 subword tokenization 对字符级任务的影响

## 局限性 / 可改进方向

1. 主要在中文 CSC 任务上验证，其他语言和字符级任务（如英文拼写纠错、字符级理解）的验证不足
2. TIPA 需要额外的训练步骤，虽然数据来自词汇表但仍需微调
3. MTIPA 的采样比例 r 的最优值需要针对不同任务调整
4. 全参数微调的成本较高，小模型上 LoRA 的效果可能有限
5. 对于非 BPE 分词的模型（如 byte-level 模型），TIPA 的适用性有待探讨

## 相关工作与启发

- **ByT5** (Xue et al., 2022)：byte-level 模型，天然有字符级精度但需改架构
- **C-LLM** (Li et al., 2024)：使用字符级 tokenization 增强字符理解
- **ReLM** (Liu et al., 2024)：将 CSC 重新定义为句子改写任务
- **逆向诅咒现象** (Berglund et al., 2023)：模型难以理解反向关系，与 TIPA 的逆序训练相呼应
- 对未来工作的启发：token 内部结构是一个被忽视的重要维度，值得更多关注

## 评分

- 新颖性: ⭐⭐⭐⭐ — 逆序字符预测思路新颖，但核心还是数据增强+SFT
- 实验充分度: ⭐⭐⭐⭐ — CSC 任务实验充分，但其他字符级任务覆盖较少
- 写作质量: ⭐⭐⭐⭐ — 方法描述清晰，新评估指标定义明确
- 价值: ⭐⭐⭐⭐ — 揭示了重要的字符级理解问题，对中文 NLP 尤为实用
