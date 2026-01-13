from scipy.stats import wilcoxon, ttest_rel
import numpy as np


def calculate_significance(results_rpcf, results_vns, ds_name="Dataset"):
    """
    Calculates statistical significance between RPCF and VNS-RPCF results
    using Wilcoxon Signed-Rank Test and Paired t-test.

    Args:
        results_rpcf (list): List of accuracy scores for RPCF across folds.
        results_vns (list): List of accuracy scores for VNS-RPCF across folds.
        ds_name (str): Name of the dataset for logging.
    """
    print(f"\n--- Statistical Significance Tests ({ds_name}) ---")

    # Ensure they are numpy arrays
    res_rpcf = np.array(results_rpcf)
    res_vns = np.array(results_vns)

    if len(res_rpcf) != len(res_vns):
        print("Error: Number of results must match for paired tests.")
        return

    # Wilcoxon Signed-Rank Test
    # Use zero_method='zsplit' or 'pratt' if needed, but default 'wilcox' often excludes zeros.
    # If all differences are zero, it will throw an error or warning.
    try:
        # Check if all values are identical
        if np.allclose(res_rpcf, res_vns):
            print("Wilcoxon: Identical results (p-value = 1.0)")
            p_wilcoxon = 1.0
        else:
            # We use 'correction=True' for continuity correction if available/needed
            stat_w, p_wilcoxon = wilcoxon(res_rpcf, res_vns)
            print(f"Wilcoxon Signed-Rank Test: p-value = {p_wilcoxon:.5f}")
    except Exception as e:
        print(f"Wilcoxon Test Failed: {e}")
        p_wilcoxon = 1.0

    # Paired t-test
    try:
        stat_t, p_ttest = ttest_rel(res_rpcf, res_vns)
        print(f"Paired t-test: p-value = {p_ttest:.5f}")
    except Exception as e:
        print(f"t-test Failed: {e}")
        p_ttest = 1.0

    # Conclusion based on alpha = 0.05 (using Wilcoxon as primary for non-parametric)
    if p_wilcoxon < 0.05:
        print(
            ">> RESULT: VNS-RPCF is statistically significantly different (p < 0.05)."
        )
    else:
        print(">> RESULT: No statistically significant difference found (p >= 0.05).")
