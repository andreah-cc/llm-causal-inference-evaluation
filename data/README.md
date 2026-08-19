# Data

## Simulated Data

The `simulated/` directory contains the synthetic datasets used in the simulation benchmark.

The project evaluates eight data-generating processes (DGPs), each designed to stress a different causal-inference challenge:

- Engine 1: linear baseline outcome, tunable confounding, constant ATE
- Engine 2: nonlinear outcome, constant ATE
- Engine 3: interaction-heavy outcome with heterogeneous treatment effects
- Engine 4: high-dimensional sparse confounding with heterogeneous treatment effects
- Engine 5: weak overlap / near-positivity violations
- Engine 6: nonlinear treatment assignment and nonlinear confounding
- Engine 7: missing outcomes under MCAR and MAR
- Engine 8: covariate measurement error and heavy-tailed outcome noise

These simulated datasets have known ground-truth treatment effects and are used to evaluate LLM-based and classical causal estimators.

## PSID Case Study

The real-world case study uses data from the **Panel Study of Income Dynamics (PSID)**, a longitudinal survey conducted by the University of Michigan.

The study follows a longitudinal design:

- **Baseline covariates:** mainly from the 2011 wave
- **Treatment:** participation in job training, vocational training, or related certification in 2013
- **Outcome:** Reference Person's 2022 labor income, reported in the 2023 PSID wave

The outcome is analyzed as `log(income + 1)`.

Baseline covariates include demographic characteristics, education, employment and work history, household structure, economic resources, public-assistance indicators, and health conditions.

After preprocessing, the final analytic sample contained:

- **6,096 observations**
- **4,240 treated units**
- **1,856 control units**

Missing treatment or outcome observations were excluded, while missing covariate values were handled using Multiple Imputation by Chained Equations (MICE).

### Why the PSID microdata are not included

Individual-level PSID data used in the analysis are **not included in this repository**.

The repository instead contains the analysis methodology, model code, derived summary results, and final report. Researchers interested in reproducing the real-data application should obtain the required public-use PSID variables through the official PSID Data Center and reconstruct the analytic dataset following the variable definitions and temporal ordering described in the final report.

### PSID variables used in this project

The PSID extracts used for this project included variables covering:

- 2011 demographics, education, employment, income, household resources, debt, and health
- 2013 job-training and vocational/certificate variables
- 2023 identifiers and Reference Person labor income for calendar year 2022

For the outcome, the project used **ER85496: LABOR INCOME OF REF PERSON-2022**.
