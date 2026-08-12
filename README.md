# Can LLMs Do Causal Inference?

## Overview

This project evaluates whether large language models (LLMs) can estimate causal effects using in-context learning (ICL).

We investigate whether a general-purpose pretrained LLM, without fine-tuning, can produce accurate and reliable Average Treatment Effect (ATE) estimates under different causal scenarios.

The framework is evaluated using:
- Synthetic datasets with known ground truth
- Classical causal inference baselines
- A real-world PSID observational data case study


## Research Questions

This project investigates:

1. Can ICL-prompted LLMs accurately estimate ATE?
2. How robust are LLM estimates under:
   - nonlinear relationships
   - heterogeneous treatment effects
   - weak overlap
   - missing data
3. How do LLM estimates compare with classical causal estimators?


## Methodology

### Simulation Benchmark

We generate datasets with controlled data-generating processes (DGPs), including:

- Linear treatment and outcome models
- Nonlinear relationships
- Treatment effect heterogeneity
- Missingness mechanisms
- Heavy-tailed noise


### Causal Estimation Methods

LLM-based estimates are compared with:

- Regression adjustment
- Inverse Probability Weighting (IPW)
- Augmented IPW (AIPW)
- Doubly Robust estimators


## Evaluation Metrics

Performance is evaluated using:

- Bias
- Root Mean Squared Error (RMSE)
- Confidence interval coverage
- Stability across prompts


## Real-world Application

A PSID observational data case study is conducted to compare LLM-based causal estimates with traditional causal inference methods.


## Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- Large Language Models
- Causal Inference
- Statistical Simulation
