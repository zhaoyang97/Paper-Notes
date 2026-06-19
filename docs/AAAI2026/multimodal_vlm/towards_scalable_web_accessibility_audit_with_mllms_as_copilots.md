---
title: >-
  [论文解读] Towards Scalable Web Accessibility Audit with MLLMs as Copilots
description: >-
  [AAAI2026][多模态VLM][web accessibility] 提出 AAA 框架，通过 GRASP（基于图的多模态页面采样）和 MaC（MLLM 作为 Copilot）两大创新，将 WCAG-EM 标准操作化，实现可扩展的端到端网页无障碍审计。 Web 无障碍性是数字包容的基础，但最新调查显示 94.8%：的百…
tags:
  - "AAAI2026"
  - "多模态VLM"
  - "web accessibility"
  - "WCAG-EM"
  - "多模态"
  - "图神经网络"
  - "page sampling"
---

# Towards Scalable Web Accessibility Audit with MLLMs as Copilots

**会议**: AAAI2026  
**arXiv**: [2511.03471](https://arxiv.org/abs/2511.03471)  
**代码**: [eaglelab-zju/AAA](https://github.com/eaglelab-zju/AAA)  
**领域**: 多模态VLM  
**关键词**: web accessibility, WCAG-EM, multimodal LLM, graph neural network, page sampling

## 一句话总结
提出 AAA 框架，通过 GRASP（基于图的多模态页面采样）和 MaC（MLLM 作为 Copilot）两大创新，将 WCAG-EM 标准操作化，实现可扩展的端到端网页无障碍审计。

## 研究背景与动机
Web 无障碍性是数字包容的基础，但最新调查显示 **94.8%** 的百万级网站首页存在 WCAG 违规。问题的根源不在于教育或工具缺失，而在于审计本身的资源瓶颈：

- **现有工具局限**：WAVE、Axe 等工具仅执行硬编码规则检查（如 alt text 缺失、对比度不足），无法覆盖语义和认知层面的问题
- **WCAG-EM 执行困难**：W3C 提出的五步审计方法论虽然标准化，但缺乏技术框架支撑大规模执行
- **页面采样不足**：现有聚类方法（SDC）仅基于浅层文本统计特征，忽略视觉布局和超链接关系等多模态语义
- **人工评估瓶颈**：手动识别无障碍关键组件（结构化页面、完整流程）需要大量专家劳动

## 方法详解

### AAA 框架整体架构
对齐 WCAG-EM 五步流程：网站爬取 → 自动检查 → 页面采样 → 人工检查 → 报告/修复。核心创新在页面采样和人工检查两阶段。

### GRASP：基于图的多模态页面采样
从三个维度定义页面代表性：
1. **文本语义代表性**：用 BERT 提取 DOM 文本的上下文化语义表示
2. **视觉布局代表性**：用 ViT 从页面截图中学习布局级视觉表示
3. **链接关系代表性**：用 GNN 在超链接图上学习结构表示

融合流程：$\mathbf{X} = \mathbf{H}_t || \mathbf{H}_v$，经 GNN message passing 后用 k-means 聚类，从每个聚类中选距中心最近的节点作为采样页面。额外引入 representativeness-enhanced graph learning，利用聚类结果修剪噪声边并恢复语义相似但未直连的边。

### MaC：MLLM 作为多角色 Copilot
- **Assistant**：自动识别 WCAG-EM 定义的结构化页面（common/relevant/essential/technology-dependent），辅助基于个体特征的页面采样；预提取无障碍关键元素（搜索栏、表单、CAPTCHA 等）
- **Auditor**：评估传统工具忽视的认知无障碍问题（WCAG 2.2 SC 3.3.8/3.3.9），如 CAPTCHA 的认知负担
- **Consultant**：提供修复建议（未来方向）

### 四个新数据集
- **TPS**：495 网站共 97,246 页面，含 DOM/截图/Axe 检查/邻接矩阵
- **APR**：968 页面，5 类网站，标注 4 类 WCAG-EM 结构化页面
- **CCT**：1,985 张 CAPTCHA 图像，17 类认证任务，评估认知无障碍
- **CPE**：1,199 页面，标注搜索/筛选/表单/CAPTCHA/联系方式 5 类组件

## 实验关键数据

### GRASP 页面采样（495 网站平均）

| 方法 | Layout $S_{sampled}$↓ | Layout $D_{intra-inter}$↑ | Text $S_{sampled}$↓ | Text $D_{intra-inter}$↑ |
|---|---|---|---|---|
| SDC_content | 56.66 | 9.96 | 89.29 | 2.73 |
| SDC_tags | 54.18 | 10.76 | 88.76 | 2.12 |
| GRASP_GCN | 51.54 | 13.05 | 86.99 | 1.59 |
| **GRASP_IGNN** | **44.31** | **14.94** | **80.45** | **7.40** |

GRASP_IGNN 在两个表示空间中均大幅领先，异质图建模更适合网站超链接结构。

### MaC 在 APR/CPE 上的 F1
- GPT-4o 在搜索栏识别上 F1=98.01%，CAPTCHA 检测 F1=95.33%
- 小模型 Qwen2.5-VL-72B 在 Relevant 页面识别上 F1=80.21%，超过 GPT-4o (35.44%)
- 认知 CAPTCHA 分类：fine-tuned Intern2-VL-8B 达 macro-F1=45.58%，超过 GPT-4o (29.16%)

## 亮点
- **首个端到端 WAA 框架**：对齐 WCAG-EM 五步流程，覆盖全审计生命周期
- **多模态页面采样**：首次整合文本、视觉、链接三维代表性，GRASP_IGNN 效果显著优于纯文本方法
- **MLLM 多角色定位**：超越评估/修复的窄范围，探索 MLLM 在采样、预审计定位、认知无障碍评估中的应用
- **小模型潜力**：实验表明 fine-tuned 8B 模型可作为领域专家，成本效益高

## 局限性
- GRASP 依赖 BERT/ViT 预训练质量，对非英文网站效果未验证
- MLLM 在 Relevant 页面识别等任务上仍有较大提升空间（GPT-4o F1 仅 35%）
- 认知 CAPTCHA 分类的 macro-F1 最高 45.58%，距实用要求仍有距离
- 数据集规模有限（APR 仅 968 页面 / 5 网站），泛化性需进一步验证

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次系统地将 MLLM 和 GNN 整合到 WCAG-EM 全流程审计中
- 实验充分度: ⭐⭐⭐⭐ — 495 网站采样实验+5 MLLM 对比+4 数据集，覆盖面广
- 写作质量: ⭐⭐⭐⭐ — 框架清晰，与标准对齐好，但细节较多
- 价值: ⭐⭐⭐⭐ — 对网页无障碍大规模审计有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Customizing Visual Emotion Evaluation for MLLMs: An Open-vocabulary, Multifaceted, and Scalable Approach](../../ICLR2026/multimodal_vlm/customizing_visual_emotion_evaluation_for_mllms_an_open-vocabulary_multifaceted_.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](../../ICLR2026/multimodal_vlm/liveweb-ie_a_benchmark_for_online_web_information_extraction.md)
- [\[AAAI 2026\] Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh](towards_human-ai_accessibility_mapping_in_india_vlm-guided_annotations_and_poi-c.md)
- [\[ICML 2026\] FlowNar: Scalable Streaming Narration for Long-Form Videos](../../ICML2026/multimodal_vlm/flownar_scalable_streaming_narration_for_long-form_videos.md)

</div>

<!-- RELATED:END -->
