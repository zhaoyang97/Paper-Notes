# D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies

**会议**: AAAI 2026  
**arXiv**: [2511.16590](https://arxiv.org/abs/2511.16590)  
**代码**: 未公开  
**领域**: Agent / LLM  
**关键词**: GUI Agent, 鲁棒性评估, 动态benchmark, 真实世界异常, 自主代理  

## 一句话总结
提出D-GARA，首个系统评估GUI Agent在真实世界异常情况（如UI变化、弹窗干扰、网络延迟、元素缺失等）下鲁棒性的动态benchmark框架，揭示现有GUI Agent在面对环境扰动时性能急剧下降的严重问题。

## 背景与动机
GUI Agent（如CogAgent、AppAgent等）在标准benchmark上表现不错，但现实环境充满各种异常情况——应用更新导致UI变化、随机弹窗、网络不稳定、权限对话框等。现有benchmark（如MiniWob、Mind2Web）使用静态网页或固定场景，无法评估Agent对这些异常的鲁棒性。这是GUI Agent实际部署的核心瓶颈。

## 核心问题
如何系统化地评估GUI Agent在真实世界异常条件下的鲁棒性？需要一个能动态注入各类异常、可控难度的benchmark框架。

## 方法详解
### 整体框架
D-GARA定义了多类真实世界异常（UI布局变化、弹窗/广告干扰、网络延迟/错误、元素缺失/变化、权限请求等），并设计可控的异常注入机制，在执行过程中动态修改GUI环境来评估Agent的适应性。

### 关键设计
1. **异常分类体系**: 系统化定义GUI Agent可能遇到的真实世界异常类型
2. **动态注入机制**: 运行时动态修改GUI环境（而非静态构造），模拟真实场景
3. **多维度评估**: 不仅看任务完成率，还评估错误恢复能力、适应速度等

## 实验关键数据
评估了多个SOTA GUI Agent在有/无异常条件下的performance gap。

## 亮点
- **填补重要空白** — GUI Agent鲁棒性评估是实际部署的关键但被忽视的问题
- **动态评估**比静态构造更接近真实场景

## 局限性 / 可改进方向
- 仅基于abstract信息，完整实验数据待补充
- 异常类型可能不够全面
- 未提出解决方案，仅诊断问题

## 与相关工作的对比
- 区别于MiniWob/Mind2Web等静态benchmark — D-GARA通过动态注入异常来评估鲁棒性
- 与Co-EPG互补 — Co-EPG提升Agent能力，D-GARA暴露Agent弱点

## 启发与关联
- GUI Agent的鲁棒性问题可能需要从训练数据入手——加入异常场景的训练样本
- 异常处理能力可能需要专门的"异常检测+修复"模块

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统化的GUI Agent鲁棒性框架
- 实验充分度: ⭐⭐⭐ 待看full paper，但框架设计系统
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐⭐ 对GUI Agent部署有重要诊断价值
