import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Drop ID if present
    if "Patient_ID" in df.columns:
        df = df.drop("Patient_ID", axis=1)

    # Standardize names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_", regex=False)
    df.columns = df.columns.str.replace("-", "_", regex=False)

    # IMPORTANT: remove duplicates created by standardization
    df = df.loc[:, ~df.columns.duplicated()]

    return df

df = load_data("Final_Balanced_Autoimmune_Disorder_Dataset (4).csv")

st.title("Autoimmune Dataset")

target_col = "Diagnosis"
if target_col not in df.columns:
    st.error("Column 'Diagnosis' not found. Please check your dataset.")
    st.stop()


binary_cols = [c for c in df.columns if df[c].nunique() == 2]

# Remove non-marker columns if they happen to be binary/categorical
remove_cols = [target_col]
if "Gender" in df.columns:
    remove_cols.append("Gender")

binary_cols = [c for c in binary_cols if c not in remove_cols]

if len(binary_cols) == 0:
    st.error("No binary marker columns found (0/1).")
    st.stop()


st.sidebar.header("Controls")

include_other = st.sidebar.checkbox("Include 'Other' diagnosis", value=False)

top_n = st.sidebar.slider("Number of top markers (Top N)", min_value=5, max_value=20, value=10, step=1)

show_values = st.sidebar.checkbox("Show values (annot)", value=True)

# Apply filter

df_plot = df.copy()
if not include_other:
    df_plot = df_plot[df_plot[target_col] != "Other"]


marker_variation = df_plot.groupby(target_col)[binary_cols].mean().std().sort_values(ascending=False)

top_markers = list(marker_variation.head(top_n).index)

heatmap_data = df_plot.groupby(target_col)[top_markers].mean()


with st.expander("See top markers ranked by variability"):
    st.write(marker_variation.head(top_n))


fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    heatmap_data,
    annot=show_values,
    ax=ax
)
ax.set_title(f"Top {top_n} Markers by Diagnosis ")
ax.set_xlabel("Marker")
ax.set_ylabel("Diagnosis")
plt.tight_layout()

st.pyplot(fig)


st.caption(
    "Each cell is the prevalence (mean of 0/1) for a marker within a diagnosis. "
    "Markers shown are selected by highest variability across diagnoses."
)
