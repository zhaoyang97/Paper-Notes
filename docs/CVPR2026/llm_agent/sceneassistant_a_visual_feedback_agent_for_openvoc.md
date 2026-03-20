# SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation

**会议**: CVPR 2026  
**arXiv**: [2603.12238](https://arxiv.org/abs/2603.12238)  
**代码**: [github.com/ROUJINN/SceneAssistant](https://github.com/ROUJINN/SceneAssistant)  
**领域**: 3D视觉 / 场景生成  
**关键词**: 3D场景生成, 开放词汇, VLM Agent, 视觉反馈, ReAct  

## 一句话总结
提出基于视觉反馈的VLM agent框架，通过14个完备Action API让VLM在ReAct闭环中迭代优化3D场景布局，无需预定义空间关系模板，在人类评估中Layout得分7.600（vs SceneWeaver 5.800），Human Preference 65%。

## 背景与动机
现有text-to-3D scene方法分两类：数据驱动方法(NeRF/3DGS)受限于数据集且不可编辑；基于LLM的检索布局方法(Holodeck/SceneWeaver)依赖预定义空间关系原语(on, face_to等)，局限于特定领域。当描述涉及预定义词汇之外的复杂空间配置时，优化过程失败。

## 核心问题
如何实现不受领域限制的开放词汇3D场景生成？不能依赖预定义空间关系模板，需要VLM自行通过视觉理解来判断和调整布局。

## 方法详解

### 整体框架
自然语言描述 → VLM agent(Gemini-3.0-Flash)按ReAct范式迭代：每步接收渲染图+物体元数据 → 推理选择Action API执行 → Blender渲染新场景 → 视觉反馈 → 循环直到Finish或最大20步。3D资产由Z-Image(文生图)+Hunyuan3D(图生3D mesh)生成。

### 关键设计
1. **完备Action API**: 14个原子操作分三类——物体增删(Create/Duplicate/Delete)、6-DoF操控(Place/Translate/Rotate/Scale)、相机控制(ViewScene/FocusOn等)。核心设计：抽象底层Blender操作为语义化命令，避免让VLM生成复杂代码而分散推理注意力。

2. **纯视觉反馈闭环**: 每步只提供当前渲染图(不累积历史图像)+历史action序列+物体坐标。渲染图上标注物体名称标签+坐标轴HUD做视觉增强。BVH-tree碰撞检测自动通知agent。

3. **自校正与质量控制**: 3D资产质量不佳时，agent可视觉发现→Delete→修改描述重新Create。地面穿透自动修正。training-free，完全基于VLM zero-shot推理。

### 损失函数 / 训练策略
无训练。纯prompt engineering驱动VLM agent行为。

## 实验关键数据
| 场景类型 | 方法 | Layout↑ | Obj Quality↑ | Human Pref↑ |
|---------|------|---------|-------------|-------------|
| Indoor(8) | Holodeck | 4.475 | 4.763 | 6.25% |
| Indoor(8) | SceneWeaver | 5.800 | 6.150 | 36.25% |
| Indoor(8) | **SceneAssistant** | **6.888** | **6.950** | **61.25%** |
| Open-vocab(22) | **SceneAssistant** | **7.600** | **7.277** | **65.00%** |

### 消融实验要点
- 去掉Action API(改JSON输出): Layout -0.595, Preference -29pp——"认知分散"
- 去掉视觉反馈(one-shot): Layout -1.345, Preference -38pp——影响最大
- 去掉Visual Prompting(标签/坐标轴): agent无法精确定位物体

## 亮点
- Action API抽象思路精妙——让VLM保持在"推理最优区间"
- 纯视觉反馈闭环，不依赖场景图等结构化中间表示
- 模块化可扩展，Action API可方便添加新操作

## 局限性 / 可改进方向
- 复杂场景下agent倾向批量初始化所有物体再调整，不够增量
- 受限于VLM/3D生成器能力天花板，需多次运行才能得到满意结果
- 评估仅human evaluation（30场景/10评估者），缺乏自动化指标

## 与相关工作的对比
- **Holodeck**: 预定义空间关系+物理求解器，Indoor Pref仅6.25%
- **SceneWeaver**: 反射式agent但仍依赖预定义空间原语，36.25%

## 启发与关联
- VLM-as-Agent的Action API设计范式可迁移到其他3D生成/编辑任务
- 视觉反馈闭环 > 开环生成的insight对所有VLM agent系统有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 纯视觉反馈+完备API的组合新颖，但ReAct框架本身不新
- 实验充分度: ⭐⭐⭐ 仅human evaluation，无自动化指标
- 写作质量: ⭐⭐⭐⭐ 清晰易懂
- 价值: ⭐⭐⭐⭐ 为开放词汇3D场景生成提供了实用范式
