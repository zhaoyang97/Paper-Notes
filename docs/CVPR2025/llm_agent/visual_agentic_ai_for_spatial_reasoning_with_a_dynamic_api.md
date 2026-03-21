# Visual Agentic AI for Spatial Reasoning with a Dynamic API

**会议**: CVPR 2025  
**arXiv**: [2502.06787](https://arxiv.org/abs/2502.06787)  
**代码**: [https://github.com/damianomarsili/VADAR](https://github.com/damianomarsili/VADAR)  
**领域**: LLM Agent  
**关键词**: 3D空间推理, 程序合成, 动态API, LLM多Agent协作, GPT-4o

## 一句话总结
提出 VADAR，一种 agentic 程序合成方法用于 3D 空间推理。多个 LLM agent 协作生成 Pythonic API 并在求解过程中动态扩展新函数来解决常见子问题，克服了 VisProg/ViperGPT 等先前方法依赖静态人工定义 API 的局限。同时引入涉及多步空间定位和推理的新 benchmark，在 3D 理解任务上超越现有零样本方法。

## 研究背景与动机
1. **领域现状**：视觉推理——解读视觉世界的能力——对操作 3D 场景的具身 agent 至关重要。VLM（如 GPT-4V、LLaVA）可以回答图像相关问题，但在涉及深度、距离、空间关系等 3D 空间推理上表现明显衰退。
2. **现有痛点**：VisProg/ViperGPT 等先前的程序合成方法依赖静态、人工定义的 API（如 `find_object()`, `compute_distance()`），限制了可处理查询的范围——不在 API 中的操作无法执行。
3. **核心矛盾**：3D 空间推理任务需要多步定位和推理（如"找到桌子→找到桌子上方的灯→找到灯左边的物体"），固定 API 无法覆盖如此多样化的查询需求。
4. **本文要解决什么？** 通过动态 API 生成，让程序合成方法能处理更广泛的 3D 空间推理查询。
5. **切入角度**：让三个专职 LLM agent（签名设计→实现→求解）协作构建和扩展 API，类似程序员逐步构建工具库的过程。
6. **核心idea一句话**：三 Agent 协作动态生成 Pythonic API（签名→DFS 实现→程序求解） + 新 Omni3D-Bench 验证多步 3D 空间推理。

## 方法详解

### 整体框架
给定图像和 3D 空间查询，三个 GPT-4o agent 协作：Signature Agent 提出新 API 函数签名（每批 10 个问题）→ API Agent 用 DFS 递归实现函数 → Program Agent 调用完整 API 生成求解程序。基础视觉模块包括 GroundingDINO+SAM2（定位）、UniDepth（深度估计）和 VLM VQA。

### 关键设计

1. **三 Agent 协作架构**:
   - **Signature Agent**：输入当前 API 签名 + 10 个问题批次 → 提出新函数签名和 docstring（如 `_get_material(image, bbox)` 或 `_is_in_front_of(image, bbox1, bbox2)`）。方法名必须以下划线开头。每批处理 10 题以避免冗余方法
   - **API Agent**：对每个新签名用 DFS 遍历依赖树递归实现——如果实现中调用了未实现的方法，先递归实现被依赖方法。失败最多重试 5 次（将错误信息反馈给 LLM），支持无限递归检测
   - **Program Agent**：用完整 API（预定义+动态生成）为每个查询生成 Python 程序，最终答案存入 `final_result` 变量
   - 设计动机：关注点分离——签名设计/实现/求解三个阶段各自优化，避免单一 agent 认知负担过重

2. **动态 API 生长机制**:
   - 预定义 API 仅包含 5 个基础模块：`loc()`（GroundingDINO+SAM2 定位）、`vqa()`（VLM 问答）、`depth()`（UniDepth 深度）、`same_object()`、`get_2D_object_size()`
   - 动态生成的 API 函数可组合基础模块，如 `_is_in_front_of()` 实现为比较两个 bbox 的 `depth()` 返回值
   - API 随问题批次逐步增长，后续同类问题可复用已有函数
   - 设计动机：VisProg/ViperGPT 的静态 API 无法覆盖多样化的 3D 推理需求

3. **Omni3D-Bench 新基准**:
   - 500 个（图像, 问题, 答案）三元组，来自 Omni3D 真实 3D 场景数据集
   - 四类问题：数值计数、数值其他（距离/尺寸/比例）、是否题、多选题
   - 非模板化查询，需多步空间定位和推理

### 训练策略
零样本——仅 in-context learning，三个 Agent 均用 GPT-4o，无需任何任务特定训练数据。

## 实验关键数据

### Omni3D-Bench（500 题，真实 3D 场景）
| 方法 | count | numeric | y/n | multi-choice | Total |
|------|:---:|:---:|:---:|:---:|:---:|
| GPT-4o（直接） | 28.1 | 35.5 | 66.7 | 57.2 | 42.9 |
| Claude 3.5-Sonnet | 22.4 | 20.6 | 62.2 | 50.6 | 32.2 |
| ViperGPT | 20.0 | 15.4 | 56.0 | 42.4 | 33.5 |
| VisProg | 2.9 | 0.9 | 54.7 | 25.9 | 21.1 |
| **VADAR** | **21.7** | **35.5** | 56.0 | **57.6** | **40.4** |
| **VADAR + oracle** | 89.5 | - | 100.0 | 94.1 | **94.4** |

### CLEVR（1155 题，合成 3D 场景）
| 方法 | Total |
|------|:---:|
| GPT-4o | 58.4 |
| ViperGPT | 26.2 |
| VisProg | 31.2 |
| **VADAR** | **53.6** |
| **VADAR + oracle** | **83.0** |

### 消融实验（Oracle 分析）
| 配置 | Omni3D-Bench | CLEVR | 说明 |
|------|:---:|:---:|------|
| VADAR | 40.4% | 53.6% | 实际性能 |
| VADAR + oracle | **94.4%** | **83.0%** | 视觉模块完美时 |
| VisProg + oracle | 66.0% | 39.9% | 对比：静态 API 天花板 |
| ViperGPT + oracle | 54.9% | 40.6% | 对比：固定函数集限制 |

### 关键发现
- **动态 vs 静态 API 的核心优势**：Oracle 模式下 VADAR 94.4% vs VisProg 66.0%（+28.4%），证明动态 API 生成在逻辑层面远优于静态 API
- 当前性能瓶颈主要在视觉模块精度（40.4%→94.4% 的差距），程序合成逻辑本身非常可靠
- VADAR 在 numeric(other) 指标上与 GPT-4o 持平（35.5%），但在非空间 GQA 上不如 GPT-4o（46.1% vs 54.9%），说明优势集中在空间推理
- 失败主要来源：GroundingDINO 检测遗漏、严重遮挡、以及 5+ 步推理链

## 亮点与洞察
- **"创造工具"而非"使用工具"**的范式升级意义重大——agent 不受人工预设的能力边界限制。这一思路本质上让 AI 系统具备了"自我扩展"能力，API 库会随使用变得越来越强大
- **动态 API 扩展思路的通用性**可迁移到任何需要 LLM 程序合成的领域——机器人控制、科学计算、数据分析等
- **多步 3D 推理 benchmark** 填补了评估空白——现有 benchmark 多为单步推理，无法衡量 agent 的组合推理能力

## 局限性 / 可改进方向
- 动态生成的 API 质量高度依赖 LLM 的代码生成能力——弱 LLM 可能生成错误或低效的函数
- API 库的无限增长需要管理机制（去重、版本控制、质量过滤），否则会引入冗余和冲突
- 多 agent 协调的通信开销和失败恢复机制需要更深入的设计
- 当前 3D 场景理解依赖于已有的感知 API（如 GroundedSAM），perception 误差会传播
- benchmark 规模相对较小，大规模验证有待开展

## 相关工作与启发
- **vs VisProg / ViperGPT**: 这些先驱方法使用静态人工定义的 API 进行程序合成，VADAR 将 API 本身变为动态可生成的
- **vs Voyager (MineCraft)**: Voyager 在 Minecraft 中也使用 LLM 动态生成技能库，但面向 2D 游戏。VADAR 将类似思路引入 3D 空间推理
- **对 Agent 领域的深远启发**: "工具创造"能力可能是通向 AGI 的关键能力之一——人类的核心优势之一就是制造和改进工具

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 动态 API 生成范式创新性强
- 实验充分度: ⭐⭐⭐⭐ 新 benchmark + 零样本对比 + 消融分析
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 工具使用范式有重要启发
