<!-- 由 src/gen_stubs.py 自动生成 -->
# MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization

**会议**: NEURIPS2025  
**arXiv**: [2510.21473](https://arxiv.org/abs/2510.21473)  
**代码**: 待确认  
**领域**: image_restoration / llm_reasoning  
**关键词**: 扩散语言模型, 多奖励优化, 推理, 重要性采样  

## 一句话总结
MRO通过多奖励优化捕获扩散语言模型内/间序列token相关性，加速DLM推理同时保持性能。

## 背景与动机
DLM推理落后于自回归LLM，需专门优化。

## 方法详解
Reject sampling + RL；组步重要性采样；多奖励加权。

## 实验关键数据
推理基准改进，采样加速（仅摘要）。

## 亮点
DLM专用多奖励优化。

## 局限性
仅摘要；方法复杂度。

## 评分
- 新颖性: ⭐⭐⭐⭐ DLM推理优化
- 实验充分度: ⭐⭐ 仅摘要
- 写作质量: ⭐⭐⭐
- 价值: ⭐⭐⭐⭐ DLM新方向
