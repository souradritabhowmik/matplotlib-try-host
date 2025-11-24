import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import io

# -------------------------- Page Config --------------------------
st.set_page_config(page_title="Distribution Fitter Pro", layout="centered")
st.title("Distribution Fitter Pro")
st.markdown("### Fit 10+ statistical distributions to your data — automatically or manually!")

# -------------------------- Available Distributions --------------------------
distributions = {
    "Normal": stats.norm,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Log-Normal": stats.lognorm,
    "Beta": stats.beta,
    "Chi-Squared": stats.chi2,
    "Uniform": stats.uniform,
    "Cauchy": stats.cauchy,
    "Laplace": stats.laplace,
    "Gumbel (Right)": stats.gumbel_r,
    "Pareto": stats.pareto,
}

# -------------------------- Sidebar: Data Input --------------------------
st.sidebar.header("1. Upload or Enter Data")
data_option = st.sidebar.radio("Choose input method", ["Enter data manually", "Upload CSV file"])

data = None

if data_option == "Enter data manually":
    manual_input = st.sidebar.text_area(
        "Enter numbers (comma, space, or newline separated)", 
        "10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 45, 50, 60, 70, 100"
    )
    try:
        data = np.array([float(x) for x in manual_input.replace(',', ' ').split() if x.strip()])
        if len(data) < 5:
            st.sidebar.warning("Please enter at least 5 data points.")
            data = None
    except:
        st.sidebar.error("Invalid input. Use numbers only.")
        data = None

else:  # Upload CSV
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] == 1:
                data = df.iloc[:, 0].dropna().values
            else:
                col = st.sidebar.selectbox("Select column", df.columns)
                data = df[col].dropna().values
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

if data is None or len(data) < 5:
    st.info("Please provide at least 5 data points to continue.")
    st.stop()

# Clean data
data = data[~np.isnan(data)]
data = data[np.isfinite(data)]
if len(data) < 5:
    st.error("Not enough valid data points after cleaning.")
    st.stop()

# -------------------------- Main Layout with Tabs --------------------------
tab1, tab2 = st.tabs(["Automatic Fitting (Best Fit)", "Manual Fitting"])

# ===================================================================
# ========================== TAB 1: Auto Fitting =====================
# ===================================================================
with tab1:
    st.header("Automatic Distribution Fitting")
    
    selected_dists = st.multiselect(
        "Choose distributions to fit (you can select multiple)", 
        options=list(distributions.keys()),
        default=["Normal", "Gamma", "Weibull", "Log-Normal", "Exponential"]
    )

    results = []
    best_dist_name = None
    best_params = None
    best_ks_stat = np.inf

    if selected_dists:
        progress_bar = st.progress(0)
        for i, dist_name in enumerate(selected_dists):
            dist = distributions[dist_name]
            try:
                with st.spinner(f"Fitting {dist_name}..."):
                    # Special case for distributions with location/scale
                    if dist_name in ["Uniform", "Cauchy", "Laplace"]:
                        params = dist.fit(data)
                    else:
                        params = dist.fit(data)
                    # Kolmogorov-Smirnov test for goodness of fit
                    ks_stat, _ = stats.kstest(data, dist.cdf, args=params)
                    results.append((dist_name, params, ks_stat, dist))
                    if ks_stat < best_ks_stat:
                        best_ks_stat = ks_stat
                        best_dist_name = dist_name
                        best_params = params
            except Exception as e:
                results.append((dist_name, None, np.inf, dist))
            progress_bar.progress((i + 1) / len(selected_dists))

        # Display results table
        st.subheader("Fitting Results (Lower KS statistic = better fit)")
        result_df = pd.DataFrame([
            {
                "Distribution": name,
                "Parameters": params if params is not None else "Failed",
                "KS Statistic": f"{ks:.5f}" if ks is not np.inf else "Failed",
                "Status": "Best" if name == best_dist_name else ""
            } for name, params, ks, _ in results
        ], index=range(1, len(results)+1))
        st.dataframe(result_df.style.highlight_min("KS Statistic", color="#90EE90"))

        # Plot best fit
        if best_dist_name:
            st.success(f"Best distribution: **{best_dist_name}**")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Histogram
            counts, bins, _ = ax.hist(data, bins=30, density=True, alpha=0.7, color="#3498db", label="Data")
            # Best fit curve
            x = np.linspace(data.min(), data.max(), 1000)
            y = distributions[best_dist_name].pdf(x, *best_params)
            ax.plot(x, y, 'r-', lw=3, label=f"{best_dist_name} fit")
            ax.set_title(f"Best Fit: {best_dist_name}", fontsize=16)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
            st.pyplot(fig)

            # Show parameters & error metrics
            st.subheader("Fit Quality Metrics")
            fitted_pdf = distributions[best_dist_name].pdf(data, *best_params)
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                relative_error = np.abs(fitted_pdf - np.histogram(data, bins=30, density=True)[0].mean())
            mae = np.mean(relative_error)
            max_error = np.max(np.abs(fitted_pdf - np.histogram(data, bins=30, density=True)[0].mean()))
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error (MAE)", f"{mae:.6f}")
            col2.metric("Max Error", f"{max_error:.6f}")
            col3.metric("KS Statistic", f"{best_ks_stat:.6f}")

# ===================================================================
# ========================== TAB 2: Manual Fitting =====================
# ===================================================================
with tab2:
    st.header("Manual Parameter Adjustment")
    manual_dist = st.selectbox("Choose a distribution for manual tuning", options=list(distributions.keys()))

    dist = distributions[manual_dist]

    # Default starting parameters from auto-fit
    try:
        default_params = dist.fit(data)
    except:
        default_params = (1.0, data.mean(), data.std())  # fallback

    # Create sliders based on number of shape parameters
    param_names = ["loc", "scale"]  # always present
    if manual_dist in ["Gamma", "Weibull", "Log-Normal", "Beta", "Chi2", "Pareto", "Gumbel"]:
        param_names.insert(0, "shape")  # shape first if needed
    if manual_dist == "Beta":
        param_names = ["a", "b", "loc", "scale"]  # beta has two shapes
    if manual_dist in ["Pareto"]:
        param_names = ["b", "loc", "scale"]

    sliders = {}
    for i, pname in enumerate(param_names):
        if pname in ["loc"]:
            sliders[pname] = st.slider(f"{pname}", float(data.min()-10, data.max()+10, default_params[i])
        elif pname in ["scale", "b", "a", "shape"]:
            sliders[pname] = st.slider(f"{pname}", 0.01, 50.0, float(default_params[i]) if i < len(default_params) else 1.0, 0.01)

    # Manual plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(data, bins=30, density=True, alpha=0.7, color="#9b59b6", label="Data")
    x_manual = np.linspace(data.min(), data.max(), 1000)
    try:
        manual_params = tuple(sliders.values())
        y_manual = dist.pdf(x_manual, *manual_params)
        ax2.plot(x_manual, y_manual, 'g-', lw=3, label=f"{manual_dist} (manual)")
        ax2.set_title(f"Manual Fit: {manual_dist}", fontsize=16)
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Density")
        ax2.legend()
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Invalid parameters: {e}")

# ===================================================================
# ========================== Footer ======================================
# ===================================================================
st.markdown("---")
st.caption("Built with Streamlit • Perfect for your class project!")
