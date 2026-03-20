# GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs

**会议**: ICLR2026  
**arXiv**: [2505.17653](https://arxiv.org/abs/2505.17653)  
**代码**: [GitHub](https://github.com/LiAuto-DSR/GeoGramBench)  
**领域**: llm_reasoning  
**关键词**: 几何推理, 程序转几何, Benchmark, 空间推理, Asymptote代码  

## 一句话总结
提出Program-to-Geometry任务和GeoGramBench(500题)，用三级几何复杂度分类法(基元识别/局部组合/全局抽象)评估19个前沿LLM从程序代码构建几何表征并推理的能力，发现所有模型在最高抽象级别准确率均低于50%。

## 背景与动机
1. 几何空间推理是AI的基础能力(机器人/自动驾驶/自动设计)，但LLM从过程式代码进行几何推理的能力被严重低估
2. 现有benchmark(MATH-500/AIME24)包含少量Asymptote代码题，但缺乏系统性评测
3. DeepSeek-R1在含代码的几何题(ℙ_TC)上相比纯文本题(ℙ_T)准确率骤降23.5%(AIME24)和10.9%(MATH-500)
4. 现有分类法基于推理难度(高中→竞赛)，而非几何结构复杂度——后者才是代码→几何任务的核心挑战
5. 答案泄露问题(代码中直接/间接包含答案)未被充分关注
6. 缺少专门评估从符号化过程式代码到空间几何理解的benchmark

## 方法详解
**任务定义**：Program-to-Geometry——模型解读过程式绘图代码(Asymptote/Matplotlib)构建几何表征，再进行数学推理获得答案(长度/面积/体积/角度/比率/计数)。

**三级分类法(按几何复杂度)**：
- **Primitive Recognition**：仅含1-2个几何基元(点/线/弧/圆/多边形)，关注基本性质
- **Local Relation Composition**：多个局部几何元素，需组合空间关系
- **Global Abstract Integration**：涉及3D对象/旋转/折叠/投影/递归等，需全局空间推理

**Benchmark构建**：从905K题中筛选→1,247含Asymptote代码的几何题→547题(格式标准化)→392题(去污染+答案泄露防护+正确性验证)→补充AIME24/MATH-500/Mathverse至500题。答案泄露防护：直接泄露→坐标重缩放；间接泄露→参数修改/遮蔽。

## 实验关键数据
| 模型 | Primitive | Compositional | Abstract | 总体 |
|------|-----------|--------------|----------|------|
| GPT-5 | 90.44% | 84.59% | **39.26%** | 75.01% |
| Qwen3-235B | 89.09% | 79.12% | **49.05%** | 74.00% |
| GPT-o1 | 85.92% | 76.12% | **44.67%** | 70.92% |
| DeepSeek-R1 | 83.16% | 69.07% | **36.75%** | 64.63% |

- 所有19个模型在Abstract级别均<50%准确率
- 最难子类型：Primitive/Compositional级角度题；Abstract级面积和体积题
- 从Primitive到Abstract准确率平均下降40+个百分点

## 亮点
- 形式化定义Program-to-Geometry任务，填补评测空白
- 三级几何复杂度分类比推理难度分类更适合本任务(有实证支持)
- 系统化处理答案泄露问题(直接+间接)，提高benchmark可靠性
- 19个模型的大规模评测，包含最新GPT-5和Qwen3
- 识别出3D几何(面积/体积)是当前LLM的关键瓶颈

## 局限性 / 可改进方向
- 仅500题，规模有限，各级别分布可能不够均衡
- 仅评估text-only LLM，未纳入多模态模型(可将代码渲染为图像)
- 仅关注Asymptote和Matplotlib两种绘图语言
- 未提出针对性的模型改进方案，主要是诊断性工作
- 部分题目来自公开数据集，即使做了去污染，仍有数据污染风险

## 与相关工作的对比
- 相比MathVerse/GeoSense等视觉几何benchmark，聚焦过程式代码而非图像
- 相比SGP-Bench等SVG理解benchmark，聚焦数学几何推理而非图形识别
- 相比MATH-500/AIME24中的少量代码题，GeoGramBench更系统且处理了答案泄露
- 分类法创新：按几何复杂度而非数学推理步骤分级

## 评分
- 新颖性: ⭐⭐⭐⭐ (任务定义和分类法有新意)
- 实验充分度: ⭐⭐⭐⭐⭐ (19模型覆盖面广，分析细致)
- 写作质量: ⭐⭐⭐⭐ (结构清晰RQ引导)
- 价值: ⭐⭐⭐⭐ (揭示LLM在几何代码推理上的系统性弱点)
