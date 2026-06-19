---
title: >-
  [论文解读] Functionality Understanding and Segmentation in 3D Scenes
description: >-
  [CVPR 2025][3D视觉][功能性理解] Fun3DU 首次提出针对 3D 场景功能性理解的方法，通过 LLM 链式思维解析任务描述 + VLM 多视角分割功能性物体 + 2D-3D 投票聚合，在 SceneFun3D 上大幅超越开放词汇 3D 分割基线（mIoU +13.2）。 领域现状：3D 场景理解主要聚焦于语…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "功能性理解"
  - "3D场景分割"
  - "视觉语言模型"
  - "零样本推理"
  - "链式思维"
---

# Functionality Understanding and Segmentation in 3D Scenes

**会议**: CVPR 2025  
**arXiv**: [2411.16310](https://arxiv.org/abs/2411.16310)  
**代码**: [https://tev-fbk.github.io/fun3du/](https://tev-fbk.github.io/fun3du/)  
**领域**: 3D视觉  
**关键词**: 功能性理解, 3D场景分割, 视觉语言模型, 零样本推理, 链式思维

## 一句话总结
Fun3DU 首次提出针对 3D 场景功能性理解的方法，通过 LLM 链式思维解析任务描述 + VLM 多视角分割功能性物体 + 2D-3D 投票聚合，在 SceneFun3D 上大幅超越开放词汇 3D 分割基线（mIoU +13.2）。

## 研究背景与动机

**领域现状**：3D 场景理解主要聚焦于语义/实例分割——识别常见家具物体（桌子、椅子、柜子）。近期的开放词汇 3D 分割方法（OpenMask3D、LERF、OpenIns3D）能根据自然语言描述在 3D 场景中定位物体。

**现有痛点**：功能性理解与常规物体分割有本质不同——给定任务描述如"打开天花板灯"，系统需要理解要操作的是灯开关（未在描述中提及），然后在 3D 场景中定位这个开关。这需要同时具备世界知识（推理功能性物体）和精细空间感知（定位把手、按钮、旋钮等小型交互元素）。现有开放词汇方法严重偏向分割大型家具，对小型功能性物体完全失效。

**核心矛盾**：开放词汇 3D 分割方法依赖在 3D 数据集（ScanNet 等）上预训练的 3D 提议模块，这些数据集偏向大物体，因此模型对小型交互元素（把手、旋钮、按钮）几乎没有识别能力。同时，任务描述通常不直接提及要分割的功能性物体，需要推理。

**本文目标** 如何在不需要任务特定训练的情况下，根据自然语言任务描述在真实 3D 场景中定位和分割功能性交互元素？

**切入角度**：3D 数据太少无法训练理解功能性的模型，但 2D 预训练模型（VLM）拥有丰富的世界知识和精细视觉感知能力。利用多个预训练 2D 模型组合（LLM 理解任务、VLM 定位物体、SAM 分割），在 2D 视角上完成识别后投射回 3D 点云。

**核心 idea**：用 LLM 链式思维推理出功能性物体名称，在精选视角上用 VLM 指向并分割功能性物体，通过多视角投票聚合到 3D 点云，全程零训练。

## 方法详解

### 整体框架
输入为场景点云、多视角 RGBD 图像和任务描述。四个模块流水线处理：(1) LLM 解析任务描述提取功能性物体 F 和上下文物体 O；(2) 开放词汇分割器在所有视角中定位上下文物体 O 并选择最佳视角子集；(3) VLM 在选定视角中定位并分割功能性物体 F；(4) 将 2D 分割结果通过 2D-3D 对应关系投射到点云并进行多视角投票。

### 关键设计

1. **任务描述理解（Chain-of-Thought Reasoning）**:

    - 功能：从自然语言任务描述中推断出需要分割的功能性物体 F 和包含它的上下文物体 O
    - 核心思路：用 LLM（Llama3.1-9B, 4-bit量化）通过 Chain-of-Thought 推理。系统提示设定 LLM 角色为辅助机器人操控器的助手，并提供可执行动作列表作为"停止准则"。先要求 LLM 列出完成任务的动作序列（避免抽象层次不对），再从中提取 F（如"门把手"）和 O（如"柜子"）的层级关系
    - 设计动机：直接查询 LLM 存在两个问题——(a) 抽象层次不确定（"开门"可能返回"门"而非"把手"）；(b) 上下文物体可能幻觉（正确识别"把手"但配对"柜子"而非"门"）。动作序列提供了明确的停止准则，层级关系避免了上下文错配

2. **基于评分的视角选择（Score-based View Selection）**:

    - 功能：从数千个视角中筛选出上下文物体可见性最好的少量视角（~50个），提高功能性物体分割的准确性和效率
    - 核心思路：先用 OWLv2 + RobustSAM 在所有视角中分割上下文物体 O。对每个分割掩码计算三个分数：分割置信度 $S_m$、距离分布均匀性 $S_d$（掩码离图像中心近→高分）、角度分布均匀性 $S_\alpha$（掩码点均匀分布在中心周围→高分）。距离和角度通过与参考均匀分布的 KL 散度计算：$S_d = 1 - D_{KL}(P_d || U_d)$。综合分数 $S_O = \lambda_m S_m + \lambda_d S_d + \lambda_\alpha S_\alpha$，选 top-50 视角
    - 设计动机：数千个视角中大部分不包含目标物体或视角质量差，全部处理既浪费算力又引入噪声。基于极坐标分布的评分能识别出物体居中且完整可见的最佳视角

3. **VLM 引导的功能性物体分割**:

    - 功能：在选定视角中精确定位和分割功能性物体
    - 核心思路：用 Molmo VLM 查询"Point to all the F in order to D"（如"Point to all the handles in order to open the bottom drawer"），VLM 返回图像上的点坐标，将这些点作为 SAM 的 prompt 生成精细分割掩码。查询中同时包含 F 和完整任务描述 D，让 VLM 能区分语义一致但任务不相关的物体
    - 设计动机：相比只提供功能性物体名称，包含完整任务描述能让 VLM 根据上下文消歧——例如只说"把手"可能分割到邻近家具的把手，加上"TV 下面的柜子"能限制分割范围

### 损失函数 / 训练策略
Fun3DU 是完全零训练方法，所有使用的模型（Llama3.1、OWLv2、RobustSAM、Molmo、SAM）都是冻结的预训练模型，不需要任何微调。多视角投票聚合：每个 3D 点的分数 $s_i = \sum_{k=1}^K |{p^k \text{ s.t. } \Gamma^k(p^k)=c_i}|$，即统计所有视角中映射到该点的功能性物体 mask 像素数量，归一化后以阈值 $\tau=0.7$ 生成最终 3D 掩码。

## 实验关键数据

### 主实验

**SceneFun3D split0（30 场景）**:

| 方法 | mAP | AP50 | AP25 | mAR | AR50 | AR25 | mIoU |
|------|-----|------|------|-----|------|------|------|
| **Fun3DU** | **7.6** | **16.9** | **33.3** | 27.4 | **38.2** | 46.7 | **15.2** |
| OpenMask3D | 0.2 | 0.2 | 0.4 | 20.3 | 24.5 | 27.0 | 0.2 |
| OpenIns3D | 0.0 | 0.0 | 0.0 | 40.5 | 46.7 | **51.5** | 0.1 |
| LERF | 0.0 | 0.0 | 0.0 | 34.2 | 35.1 | 36.0 | 0.0 |

**SceneFun3D split1（200 场景）**:

| 方法 | mAP | AP50 | AP25 | mIoU |
|------|-----|------|------|------|
| **Fun3DU** | **6.1** | **12.6** | **23.1** | **11.5** |
| OpenMask3D | 0.0 | 0.0 | 0.0 | 0.1 |
| OpenIns3D | 0.0 | 0.0 | 0.0 | 0.1 |
| LERF | 0.0 | 0.0 | 0.0 | 0.0 |

Fun3DU 在 AP25 上超越最近对手 OpenMask3D 达 32.9 个点（split0）和 23.1 个点（split1）。

### 消融实验

| 配置 | 说明 |
|------|------|
| 所有基线方法 | AP 近零但 AR 较高 → 严重欠分割，倾向于分割整个家具而非功能性小部件 |
| Fun3DU split0 vs split1 | split1 性能下降 → 场景更复杂（点云最大 1300万 vs 800万点） |
| OpenMask3D split1 大幅下降 | 主要依赖 3D 编码器，受场景复杂度影响更大 |
| Fun3DU/LERF/OpenIns3D 相对稳定 | 在 2D 视角上做分割对场景复杂度更鲁棒 |

### 关键发现
- **所有开放词汇 3D 分割基线几乎完全失败**——AP 近零说明它们无法分割小型功能性物体，只能分割大型家具。这证实了功能性分割需要专门的方法设计
- OpenIns3D/LERF 有较高 AR 但零 AP——它们能"召回"到附近区域但分割结果不精确（整个柜子而非把手）
- VLM（Molmo）的点指向能力是关键——它能理解任务语义并精确定位小物体
- Chain-of-Thought 推理有效避免了 LLM 的两个常见失败模式（抽象层次错误和上下文幻觉）

## 亮点与洞察
- **问题定义的重要性**：功能性理解是一个被忽视但极其重要的任务——对具身AI来说，找到"开关"比找到"柜子"更有操作意义。SceneFun3D 基准的任务描述设计（不直接提及目标物体）增加了真实性和挑战性
- **2D 能力绕过 3D 限制**：3D 数据和模型对小物体的偏差是系统性的，通过在 2D 做分割再投射到 3D 的策略优雅地绕过了这个瓶颈
- **视角选择的极坐标评分**：用距离和角度分布的均匀性来评估视角质量——物体居中且均匀分布意味着可见性好，这个评分方式新颖且直觉合理

## 局限与展望
- mIoU 最高仅 15.2%——虽然远超基线但绝对性能仍然较低，功能性分割仍有很大提升空间
- 完全依赖预训练模型的级联管线，每一步的误差会累积（LLM 推理错误→视角选错→分割失败）
- 需要处理 50 个视角的 VLM 查询，计算成本仍不低
- 仅在室内场景评测，对室外/工业场景的适用性未验证
- 对空间描述（如"上面的抽屉"vs"下面的抽屉"）的处理依赖 VLM 的空间推理能力，可能在复杂布局中失败

## 相关工作与启发
- **vs OpenMask3D/OpenIns3D**: 开放词汇 3D 分割方法依赖 3D 提议模块，偏向大物体，无推理能力。Fun3DU 通过 LLM 推理+VLM 在 2D 定位小物体解决了这两个根本限制
- **vs LERF**: LERF 基于 NeRF 的语言场，需要逐场景训练且对精细物体定位能力弱。Fun3DU 是零训练且在 2D 层面精确定位
- **vs 2D VLMs (LLaVA/Molmo)**: Molmo 的点指向能力使其特别适合功能性物体定位，优于 LLaVA 等只能文本回答的 VLM

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义并解决 3D 功能性理解问题，方法设计针对性强
- 实验充分度: ⭐⭐⭐⭐ 在专用基准上全面评测，但缺少更多消融分析
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清晰，四模块管线逻辑流畅
- 价值: ⭐⭐⭐⭐⭐ 对具身AI和人机交互领域有直接意义，填补了3D功能性理解的空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](../../ICCV2025/3d_vision/articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)
- [\[CVPR 2025\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)
- [\[CVPR 2025\] SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes](spectromotion_dynamic_3d_reconstruction_of_specular_scenes.md)
- [\[CVPR 2025\] Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)

</div>

<!-- RELATED:END -->
