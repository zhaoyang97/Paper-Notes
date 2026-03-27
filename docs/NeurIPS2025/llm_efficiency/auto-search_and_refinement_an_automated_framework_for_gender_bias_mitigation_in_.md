# Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs

**会议**: NeurIPS 2025
**arXiv**: [2502.11559](https://arxiv.org/abs/2502.11559)
**代码**: [GitHub](https://github.com/SavannahXu79/FaIRMaker)
**领域**: LLM效率 / 公平性
**关键词**: 性别偏见缓解, 自动提示搜索, Fairwords, 去偏见, LLM公平性

## 一句话总结
提出 FaIRMaker 框架，通过"自动搜索+精化"范式先用梯度优化找到去偏见触发词（Fairwords），再训练 seq2seq 模型将其转化为可读指令，在开源和闭源 LLM 上有效缓解性别偏见同时保持甚至提升任务性能。

## 研究背景与动机
1. **领域现状**：LLM 预训练中编码了社会偏见（尤其是性别偏见），现有缓解方法包括参数修改（微调/模型编辑）和指令引导（手动设计去偏见 preamble）两大类。
2. **现有痛点**：参数修改方法资源密集、不适用于闭源模型；手动设计的去偏见指令（如 counterfactual preamble CF-D）虽然降低偏见，但会显著损害正常任务性能——因为插入的性别相关描述干扰了模型对原始查询的理解。
3. **核心矛盾**：自动化梯度搜索方法（如 GCG trigger）可以在更大的搜索空间中找到有效的去偏见 token，但生成的是无意义 token 序列，不可解释且不可迁移到闭源模型。
4. **本文要解决什么？** 设计一个同时满足三个条件的方法：(1) 自动化，(2) 兼容开源和闭源模型，(3) 保持正常任务性能。
5. **切入角度**：将梯度搜索和手动设计的优势结合——先自动搜索触发词（大搜索空间），再用 seq2seq 模型精化为可读指令（可迁移性）。
6. **核心idea一句话**：用 GCG 自动搜索去偏见 trigger token，再训练 refiner 将其转化为自然语言指令。

## 方法详解

### 整体框架
FaIRMaker 包含两个阶段：Auto-Search （在偏好数据集上用 GCG 优化去偏见 Fairwords + 过滤有效的 Fairwords 进入 Fairwords Bag）→ Refinement（用 ChatGPT 做 reverse inference 将 Fairwords 精化为可读指令，训练 seq2seq refiner 模型以泛化精化过程）。推理时，随机选一个 Fairwords 和用户 query 一起输入 refiner，输出精化指令再送入 LLM。

### 关键设计

1. **Fairwords 自动搜索与过滤**:
   - 做什么：在偏好数据集上用 GCG 优化器搜索通用触发词，使其附加到性别相关查询后能提升 chosen response 概率并降低 rejected response 概率
   - 核心思路：损失函数 $s^* = \min_s -\log f_\theta(y_c|s \oplus x) + \alpha \log f_\theta(y_r|s \oplus x)$，用贝叶斯优化平衡促进公平回答和抑制偏见回答
   - 过滤：用 llama3.1-8b-instruct 作为评估器，在验证集上比较有无 Fairwords 的回答质量和偏见水平，只保留有效的 Fairwords
   - 设计动机：自动搜索解决手动设计 prompt 覆盖面有限的问题；过滤确保优化后的 Fairwords 真正有效

2. **ChatGPT 辅助精化**:
   - 做什么：用 ChatGPT 进行"逆向推理"，分析 Fairwords 为何有效，将其转化为人类可读的指令
   - 核心思路：对偏见任务和正常任务分别精化——偏见任务的精化产生去偏见指令，正常任务的精化产生提升回答质量的指令，各 9K 样本共 18K
   - 设计动机：保持梯度搜索的大搜索空间优势，同时获得手动指令的可读性和可迁移性

3. **Seq2seq Refiner 训练**:
   - 做什么：训练 Llama3.2-3b-instruct 作为 refiner，自动将 Fairwords + query 转化为精化指令
   - 核心思路：训练损失 $\mathcal{L} = -\frac{1}{N}\sum_{t=1}^N \log \mathcal{F}_{refine}(p|s \oplus x)$，使 refiner 对偏见查询生成去偏见指令，对正常查询生成保持性能的指令
   - 设计动机：使 FaIRMaker 成为独立模块，无需访问 LLM 参数，推理时自适应生成不同类型的指令

### 损失函数 / 训练策略
- Fairwords 搜索：GCG 优化器，在 GenderAlign 偏好数据集上优化
- Refiner 训练：标准 seq2seq 损失，18K 混合数据（9K 偏见 + 9K 正常）

## 实验关键数据

### 偏见缓解（BBQ-gender）
| 模型 | sDIS (原始→FM) | sAMB (原始→FM) |
|------|----------------|----------------|
| Llama2-Alpaca | 1.066→**0.224** | 0.804→**0.157** |
| Llama2-Chat | 2.233→**0.273** | 1.673→**0.189** |
| Qwen2-Instruct | 4.638→**1.906** | 1.377→**0.320** |
| Qwen2.5-Instruct | 1.212→**0.431** | 0.030→**0.012** |

### 任务性能保持（GA-test 对话质量）
| 模型 | 原始 | FaIRMaker | CF-D | Desc-D |
|------|------|-----------|------|--------|
| Llama2-Alpaca | 3.71 | **4.07** | 3.50↓ | 3.32↓ |
| Llama2-Chat | 4.56 | **4.73** | 4.00↓ | 4.00↓ |
| GPT3.5-turbo | 4.73 | **4.90** | 4.68↓ | 4.65↓ |

### 消融实验
| 配置 | 偏见缓解 | 任务性能 |
|------|---------|---------|
| 完整 FaIRMaker | 最佳 | 最佳 |
| 仅 Fairwords（无精化） | 偏见下降但不稳定 | 可能损害 |
| 仅手动指令 | 偏见下降适中 | 显著损害 |

### 关键发现
- FaIRMaker 在所有模型上一致性地降低偏见 50%+，同时提升对话质量
- 传统手动 preamble（CF-D、Desc-D）在降低偏见的同时显著损害任务性能
- Fairwords 的有效性可能与其表达的情感正相关——积极情感的 Fairwords 更有效
- 兼容闭源模型 GPT-3.5-turbo，去偏见效果一致

## 亮点与洞察
- **搜索+精化范式**：巧妙地将白盒方法的搜索能力和黑盒方法的可迁移性结合，避免了二者各自的局限
- **自适应 refiner**：能根据输入查询类型自动切换行为——偏见查询生成去偏见指令，正常查询生成提质指令；这种"情境感知"比固定 preamble 灵活得多
- **Fairwords 可解释性分析**：发现自动搜索到的有效 trigger 与"积极情感"相关，为理解去偏见机制提供了新视角

## 局限性 / 可改进方向
- **仅处理性别偏见**：未评估种族、年龄等其他类型偏见的缓解效果
- **Fairwords 搜索依赖白盒模型**：虽然推理时不需要白盒访问，但搜索阶段需要（在 Llama2 上执行），限制了 Fairwords 的多样性来源
- **评估指标局限性**：win-tie-loss 依赖 LLM-as-judge，可能存在评估偏差

## 相关工作与启发
- **vs CF-D/Desc-D（手动 preamble）**: FaIRMaker 避免了将性别相关 preamble 直接注入 prompt 导致的查询理解干扰
- **vs Sheng et al. (2020) 自动 trigger**: 那些 trigger 不可读、不可迁移；FaIRMaker 通过精化步骤解决了这两个问题
- **vs MBIAS（后处理）**: FaIRMaker 在输入端引导，不需要修改输出，更灵活

## 评分
- 新颖性: ⭐⭐⭐⭐ 搜索+精化范式有创新性，将对抗 trigger 转化为可读指令的想法新颖
- 实验充分度: ⭐⭐⭐⭐ 5个LLM、多个benchmark、详细消融
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰，对比公平，例子直观说明问题
- 价值: ⭐⭐⭐⭐ 实用的偏见缓解工具，模型无关设计增加应用性
