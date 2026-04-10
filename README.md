# LSTEM: Lightweight and Scalable Trust Evaluation Model

## What is LSTEM?

LSTEM is a trust-aware machine learning framework for IoT attack detection. It extends traditional traffic-feature-based detection by adding a dynamic trust score for each device. This trust score is calculated from observable network behavior and then integrated into the learning pipeline as an additional signal for classification.

## Why LSTEM?

IoT networks are exposed to frequent cyberattacks, while many devices have limited compute, memory, and energy. LSTEM is designed to improve detection quality without requiring heavy infrastructure. The key motivation is that raw traffic features alone may miss behavioral reliability patterns, while trust-aware features can provide better separation between benign and malicious activity.

## Overview

LSTEM evaluates device behavior using four trust components:

1. Packet Success Rate
2. Behavior Consistency
3. Protocol Compliance
4. Temporal Stability

These components are combined into a single trust score and used with existing IoT traffic features. The resulting trust-enriched feature set is then used for model training and evaluation to compare trust-aware performance against baseline approaches.


