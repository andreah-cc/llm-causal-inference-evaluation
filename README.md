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


## Results

The evaluation framework compares LLM-based causal estimates with classical causal estimators across eight simulation engines with known ground-truth Average Treatment Effects (ATEs).

The main findings are:

- Method-inducing in-context learning (ICL) substantially improves LLM estimation relative to raw-data prompting.
- In low-effect settings, the structured LLM protocols achieve accuracy comparable to classical estimators such as regression adjustment and AIPW.
- As the true treatment effect increases, the LLM tends to exhibit increasing negative bias and underestimation.
- The LLM generally reports wider confidence intervals than classical estimators, resulting in conservative uncertainty quantification and frequent over-coverage.
- In the nonlinear-confounding setting represented by Engine 6, the structured LLM protocol performs particularly well, producing lower bias and substantially better coverage than the classical benchmark used in the study.

The complete simulation summary is available in:

`results/llm_evaluation_protocol_results.xlsx`

For the full methodology, figures, and discussion, see:

`report/final_report.pdf`

## PSID Real-World Case Study

The project also applies the same causal-inference framework to the **Panel Study of Income Dynamics (PSID)**.

The empirical question is whether participation in job training, vocational training, or related certification in **2013** affects long-run labor income measured in the **2023 PSID wave**.

The analysis uses:

- pre-treatment covariates measured mainly in **2011**
- treatment measured in **2013**
- Reference Person labor income for calendar year **2022**, reported in the **2023 wave**

The final cleaned sample contains **6,096 observations**, including **4,240 treated** and **1,856 control** units.

### Classical estimates

| Estimator | ATE | SE | 95% CI |
|---|---:|---:|---:|
| Difference-in-means | -0.0047 | 0.0302 | [-0.0640, 0.0545] |
| OLS adjustment | 0.0159 | 0.0370 | [-0.0566, 0.0883] |
| PS stratification | 0.0390 | 0.0482 | [-0.0556, 0.1336] |
| AIPW / Doubly robust | 0.0339 | 0.0438 | [-0.0520, 0.1197] |

All classical 95% confidence intervals include zero, so the analysis does not provide statistically significant evidence of a nonzero long-run treatment effect under the study's specification.

### LLM protocol results

Repeated LLM runs also produce protocol-level mean estimates close to zero:

| Protocol | Mean ATE | SD ATE | Mean CI Width |
|---|---:|---:|---:|
| A: Difference-in-means | -0.031 | 0.030 | 0.268 |
| B: Regression adjustment | 0.060 | 0.071 | 0.337 |
| C: PS stratification | -0.016 | 0.118 | 0.287 |
| D: AIPW | -0.012 | 0.090 | 0.560 |

Protocol D produces the widest intervals and therefore the most conservative uncertainty reporting among the four LLM protocols.

The PSID application is intended as a real-world validation of the simulation findings. Because the true causal effect is unknown in observational data, the analysis focuses on overlap, agreement with classical estimators, and whether the LLM produces appropriately cautious conclusions.

> Note: individual-level PSID microdata are not included in this repository. See `data/README.md` for the data description and reconstruction notes.
