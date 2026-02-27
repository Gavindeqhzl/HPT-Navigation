# Socially Compliant Multi-robot Crowds Navigation via Hierarchical Preference Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the IROS 2026 paper: **"Socially Compliant Multi-robot Crowds Navigation via Hierarchical Preference Learning"**.

## 🎥 Dynamic Simulation Demonstration

To comprehensively illustrate the continuous spatiotemporal interactions and robust collision avoidance capabilities of our proposed framework, please watch our dynamic simulation demonstration below. 

*(Click the image below to watch the full video on YouTube)*

[![Simulation Demo](https://img.youtube.com/vi/5wb0veHJvoM/0.jpg)](https://youtu.be/5wb0veHJvoM)

## 📖 Abstract
Offline Multi-Agent Reinforcement Learning (MARL) offers a promising paradigm for multi-robot cooperative navigation in densely populated environments. However, formulating an explicit joint reward function that accurately encapsulates team-level social compliance remains a critical bottleneck. While Preference-based RL (PbRL) bypasses rigid manual reward engineering, applying holistic, team-level preference evaluations introduces a severe granularity mismatch, fundamentally exacerbating the multi-agent credit assignment problem. To bridge this divide, this paper proposes a unified hierarchical reward modeling framework that systematically translates sparse global preferences into dense, actionable local signals. 

The core innovation is twofold: first, a **Cascaded Social Preference (CSP)** mechanism is introduced to automatically generate consistent preference labels via prioritized evaluation criteria, eliminating the prohibitive cost of manual annotation. Second, a **Hierarchical Preference Transformer (HPT)** is designed to capture complex spatiotemporal interaction dependencies and perform dynamic spatial credit assignment. By explicitly aligning individual agent contributions with overarching global objectives, this architecture intrinsically resolves the credit assignment ambiguity, mitigates subjective reward bias, and systematically quantifies broad social norms. Extensive experiments integrating the proposed framework with multiple offline MARL algorithms demonstrate consistent improvements over conventional baselines, achieving superior performance in balancing safe collision avoidance with robust formation coordination.

## ⚙️ Quick Start

**1. Clone the repository:**
```bash
git clone [https://github.com/Gavindeqhzl/HPT-Navigation.git](https://github.com/Gavindeqhzl/HPT-Navigation.git)
cd HPT-Navigation
