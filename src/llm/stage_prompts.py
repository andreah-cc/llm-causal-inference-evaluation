"""Module 3 prompt renderers for Stage 3, Stage 4, and Stage 5."""

def render_stage3_prompt(protocol_c_input: dict) -> str:
    lines = []
    for b in protocol_c_input["bin_summary"]:
        lines.append(
            f"bin{b['bin'] + 1}: "
            f"n = {b['n']}, n1 = {b['n1']}, n0 = {b['n0']}, "
            f"meanY1 = {b['meanY1']}, sdY1 = {b['sdY1']}, "
            f"meanY0 = {b['meanY0']}, sdY0 = {b['sdY0']}"
        )
    bin_block = "\n".join(lines)

    return f"""TASK: Estimate ATE using propensity score stratification with K bins.

Return ONE JSON object with fields:
ate_hat, ci95, method, assumptions, overlap_warning, notes.

ASSUME:
- unconfoundedness_given_X
- overlap

INSTRUCTIONS:
1. For each bin k, compute tau_k = meanY1_k - meanY0_k.
2. Compute ate_hat = sum_k (n_k / n_total) * tau_k.
3. Set overlap_warning = true if any bin has n1_k = 0 or n0_k = 0.
4. In notes, briefly state whether overlap appears adequate across bins.
5. If confidence interval inputs are not available, return ci95 as null.

DATA (K = {protocol_c_input['K']}):
{bin_block}

OUTPUT JSON ONLY."""

def render_stage4_prompt(protocol_ipw_input: dict) -> str:
    return f"""TASK: Compute Hajek IPW ATE using the provided weighted sums.

Return ONE JSON object with fields:
ate_hat, ci95, method, assumptions, overlap_warning, notes.

ASSUME:
- unconfoundedness_given_X
- overlap

INSTRUCTIONS:
1. Compute treated_wmean = S1Y / S1.
2. Compute control_wmean = S0Y / S0.
3. Compute ate_hat = treated_wmean - control_wmean.
4. Set overlap_warning = true if:
   - max_weight > 20, OR
   - ESS_treated < 50, OR
   - ESS_control < 50.
5. In notes, briefly comment on whether the estimate appears stable or sensitive to weak overlap.
6. If confidence interval inputs are not available, return ci95 as null.

DATA:
S1Y = {protocol_ipw_input['S1Y']}
S1 = {protocol_ipw_input['S1']}
S0Y = {protocol_ipw_input['S0Y']}
S0 = {protocol_ipw_input['S0']}
max_weight = {protocol_ipw_input['max_weight']}
ESS_treated = {protocol_ipw_input['ESS_treated']}
ESS_control = {protocol_ipw_input['ESS_control']}

OUTPUT JSON ONLY."""

def render_stage5_prompt(stage5_input: dict) -> str:
    return f"""TASK: Choose the most appropriate treatment-effect estimator based on the diagnostics below.

Available methods:
- stratification
- IPW-H
- overlap weighting
- trimming + stratification

Return ONE JSON object with fields:
ate_hat, ci95, method, assumptions, overlap_warning, notes.

ASSUME:
- unconfoundedness_given_X
- overlap may be weak

INSTRUCTIONS:
1. If overlap is good, prefer stratification or IPW-H.
2. If overlap is weak, prefer overlap weighting or trimming + stratification.
3. Set overlap_warning = true if diagnostics indicate weak support, extreme weights, or empty bins.
4. In notes, briefly justify the estimator choice.
5. If the chosen method changes the estimand, state that explicitly.

DIAGNOSTICS:
propensity_treated_quantiles = {stage5_input['propensity_treated_quantiles']}
propensity_control_quantiles = {stage5_input['propensity_control_quantiles']}
max_weight = {stage5_input['max_weight']}
ESS_treated = {stage5_input['ESS_treated']}
ESS_control = {stage5_input['ESS_control']}
empty_bins = {stage5_input['empty_bins']}

candidate_estimates = {stage5_input['candidate_estimates']}

OUTPUT JSON ONLY."""
