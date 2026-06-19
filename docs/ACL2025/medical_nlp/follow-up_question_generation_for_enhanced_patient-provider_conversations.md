---
title: >-
  [论文解读] Follow-up Question Generation for Enhanced Patient-Provider Conversations
description: >-
  [ACL 2025][医疗NLP][追问生成] 提出 FollowupQ 多智能体框架，结合 EHR 推理、鉴别诊断和消息澄清三类 Agent，为异步医患对话自动生成个性化追问列表，在真实和半合成数据集上分别比基线提升 17% 和 5% 的 RIM 分数，将医生需要额外发送的信息收集消息减少 34%。
tags:
  - "ACL 2025"
  - "医疗NLP"
  - "追问生成"
  - "异步医疗对话"
  - "多智能体框架"
  - "电子健康记录"
  - "鉴别诊断"
---

# Follow-up Question Generation for Enhanced Patient-Provider Conversations

**会议**: ACL 2025  
**arXiv**: [2503.17509](https://arxiv.org/abs/2503.17509)  
**代码**: 未公开（数据集 FollowupBench 公开）  
**作者**: Joseph Gatto, Parker Seegmiller, Timothy Burdick, Inas S. Khayal, Sarah DeLozier, Sarah M. Preum  
**机构**: Dartmouth College  
**领域**: 医疗NLP  
**关键词**: 追问生成, 异步医疗对话, 多智能体框架, 电子健康记录, 鉴别诊断

## 一句话总结

提出 FollowupQ 多智能体框架，结合 EHR 推理、鉴别诊断和消息澄清三类 Agent，为异步医患对话自动生成个性化追问列表，在真实和半合成数据集上分别比基线提升 17% 和 5% 的 RIM 分数，将医生需要额外发送的信息收集消息减少 34%。

## 研究背景与动机

**异步医疗对话的特殊性**：
   - 在远程医疗和患者门户中，医患通过异步消息交流，不同于实时多轮对话
   - 患者常假设医生了解其背景，导致消息信息不全
   - 医生需一次性生成多个追问问题（而非逐个提问），以减少额外通信轮次

**现有方法的不足**：
   - 现有追问生成研究（社交媒体、对话调查等）多聚焦于一次生成一个问题
   - 同步对话研究未考虑异步场景的特殊需求——无法实时追问
   - 现有 LLM 直接生成追问效果差——即使让其生成 10 倍于医生数量的问题，仍无法覆盖医生实际提出的所有问题

**临床需求**：异步消息处理是医生倦怠的重要原因，自动追问生成可减轻工作负担

## 方法详解

### 整体框架

FollowupQ 是一个多智能体框架，接收患者消息 T 和 EHR 记录 C = {A, H, M}（人口信息、病史、用药），通过三类 Agent 生成追问集合 Q̂，可选地经过去重和 Top-k 筛选控制输出数量。

### 三类核心 Agent

#### 1. EHR 推理 Agent（2 个）

- **病史推理 Agent**：从患者病史 H 中提取与当前消息最相关的信息 I_hist，然后基于这些信息生成针对性追问 Q̂_hist
- **用药推理 Agent**：从用药列表 M 中提取相关药物信息 I_med，生成用药相关追问 Q̂_med
- **关键设计**：先信息提取再生成问题的两步法，避免将无关的 EHR 信息干扰追问生成

#### 2. 鉴别诊断 Agent

- **诊断生成**：分别生成最佳和最坏情况下的 k 个可能诊断 D_diff = f(T, P_best, k) ∪ f(T, P_worst, k)
- **追问生成**：对每个可能诊断 dᵢ，生成排除该诊断所需的追问 Q̂_dᵢ = f(T, dᵢ, P_rule-out, k)
- **设计动机**：模拟临床医生的思维过程——先想可能是什么病，再问能排除哪些

#### 3. 消息澄清 Agent（4 个）

- **症状询问 Agent**：提取消息中的症状，生成细化问题（如腹痛具体位置）
- **自我治疗 Agent**：询问患者正在使用的非处方药或自行治疗方式
- **时间推理 Agent**：澄清症状时间线（持续时间、频率等）
- **消息歧义 Agent**：针对模糊表述要求更多解释

### 总问题集生成

Q̂_p = Q̂_{D_diff} ∪ Q̂_{EHR} ∪ Q̂_{clar}

### 问题筛选（可选）

1. **去重**：用 LLM 识别语义重复的问题
2. **Top-k 选择**：Agent 从去重后的列表中选择 k 个最重要的问题呈现给患者

### 评估指标

1. **Requested Information Match (RIM)**：RIM(Q, Q̂) = |Q ∩ Q̂| / |Q|，衡量系统生成的问题覆盖了多少真实医生提出的问题。不惩罚额外生成的问题
2. **Message Reduction % (MR%)**：RIM = 1.0 的样本比例，表示完全覆盖医生问题的比例

### LLM-as-Judge 语义匹配

使用经微调的 PHI-4-14B 作为 Judge，判断生成问题和真实问题是否请求相同信息（即使措辞不同）。测试集上 macro F1 = 0.87。

## 实验结果

### 数据集

| 数据集 | 类型 | 样本数 | 平均问题数/样本 | 特点 |
|--------|------|:------:|:---:|------|
| FB-Real | 真实患者消息 + EHR | 150 | 3.4 | 含 PHI，不公开 |
| FB-Synth | 半合成消息 + EHR | 250 | 9.3 | 2300+ 追问，公开 |

### 主实验结果（FB-Real）

| 方法 | RIM ↑ | 平均生成问题数 |
|------|:---:|:---:|
| 0-shot (Llama3-8b) | ~0.40 | ~10 |
| Few-shot (Llama3-8b) | ~0.40 | ~12 |
| 40-question (Llama3-8b) | ~0.45 | 40 |
| Long-Thought (DeepSeek R1) | ~0.48 | ~15 |
| **FollowupQ (Llama3-8b)** | **0.62** | 36 |
| FollowupQ (Llama3-8b-Aloe) | 0.64 | >36 |

**关键发现**：FollowupQ 比零/少样本基线提升约 22 个百分点。即使让基线 LLM 生成 10 倍以上的问题（40 个 vs 医生平均 3.4 个），仍无法匹配 FollowupQ 的表现——问题不在数量而在多样性。

### 工作负担减少效果

| 方法 | MR% (RIM=1.0 比例) ↑ |
|------|:---:|
| 最好的基线 | ~15% |
| **FollowupQ (Llama3-8b)** | **34%** |

FollowupQ 在 34% 的患者消息中完全覆盖了医生的所有追问，意味着医生在这些情况下无需额外发送信息收集消息。

### 各 Agent 贡献分析（FB-Real）

| Agent 类型 | RIM 贡献 |
|------------|:---:|
| 鉴别诊断（最坏情况）Agent | 最大贡献 |
| 用药推理 Agent | ~10% |
| 时间澄清 Agent | 显著贡献 |
| 消息歧义 Agent | 显著贡献 |
| EHR 相关 Agent | ~10% |

**洞察**：最坏情况鉴别诊断贡献最多，反映了临床追问的核心动机——排除严重情况。EHR 信息贡献约 10%，证实了个性化 EHR 推理的必要性。

### 筛选后效果

- 36 个问题 → 去重后 22 个（RIM: 0.62 → 0.57）
- 去重后 22 个 → Top-10 筛选（RIM: 0.57 → 0.42）
- 下降的原因不是问题质量差，而是无法建模特定医生的个人偏好

### FB-Synth 结果

- FollowupQ (Llama3-8b) 比最近基线提升 5 个百分点
- Qwen-32b 上 FollowupQ 仍优于基线，但提升更温和

## 亮点与洞察

1. **问题定义精准**：首次系统定义异步医患追问生成任务，区别于同步对话中的信息获取
2. **临床思维建模**：三类 Agent 对应医生实际的三种思维过程（EHR 查阅、鉴别诊断、消息理解），具有临床专业的可解释性
3. **"数量不等于质量"的发现**：让 LLM 生成更多问题并不能解决问题——多样性和临床相关性更重要
4. **RIM 指标设计合理**：不惩罚额外问题，符合临床"多问不如少问遗漏"的逻辑
5. **实用性强**：34% 的消息减少率意味着真实可观的工作负担缓解

## 局限性

1. 受限于安全计算环境，仅测试了有限的 LLM（Llama3-8b/Aloe, Qwen-32b），未测试 GPT-4 等更强模型
2. 数据来自单一农村社区医院，患者群体和医生偏好可能存在偏差
3. 不同医生对同一消息可能产生不同的追问集，ground truth 具有主观性
4. Top-k 筛选效果受限于无法建模特定医生的个人偏好
5. 未探索同步对话场景的应用

## 相关工作

- **追问生成**: Meng et al. (2023) 和 Liu et al. (2025) 研究社交媒体追问，但一次只生成一个问题；Liu et al. (2024) 在极小规模（n=7）上探索了类似设置
- **医疗问诊对话**: Winston et al. (2024) 和 Li et al. (2024) 在同步对话中研究 LLM 问诊，但未考虑异步场景和 EHR 数据
- **多 Agent 医疗系统**: MedAgents (Tang et al., 2024)、RareAgents (Chen et al., 2024)、MDAgents (Kim et al., 2024) 用于医疗决策，但非信息获取

## 评分 ⭐⭐⭐⭐

- **创新性**: ⭐⭐⭐⭐ 首次系统定义异步医疗追问生成任务，多 Agent 框架设计合理
- **实验充分性**: ⭐⭐⭐⭐ 真实+合成数据集、多基线对比、Agent 贡献分析、筛选消融
- **实用价值**: ⭐⭐⭐⭐⭐ 直接解决医生异步消息过载的痛点，34% 消息减少率有实际意义
- **写作质量**: ⭐⭐⭐⭐ 问题动机和临床背景阐述清晰，评估指标设计动机充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality](arghitz_at_archehr-qa_2025_a_two-step_divide_and_conquer_approach_to_patient_que.md)
- [\[ACL 2025\] The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It](auxiliary_patient_data_xray.md)
- [\[ACL 2026\] Query Pipeline Optimization for Cancer Patient Question Answering Systems](../../ACL2026/medical_nlp/query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)
- [\[ACL 2025\] AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset](afrimed_qa_pan_african.md)
- [\[ACL 2025\] Automated Structured Radiology Report Generation](automated_structured_radiology_report_generation.md)

</div>

<!-- RELATED:END -->
