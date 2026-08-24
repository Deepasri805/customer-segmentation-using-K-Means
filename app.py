import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-header {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
}

.main-header h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}

.main-header p {
    color: #e2e8f0;
    font-size: 17px;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #e2e8f0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("customer_segmentation_data.csv")

    return df


df = load_data()


# =========================================================
# K-MEANS MODEL
# =========================================================

features = [
    "age",
    "income",
    "spending_score",
    "purchase_frequency"
]

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


kmeans_model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans_model.fit(X_scaled)


df["Cluster"] = kmeans_model.labels_


# =========================================================
# CLUSTER NAMES
# =========================================================

cluster_names = {
    0: "Budget Frequent Buyers",
    1: "Value Spenders",
    2: "Premium Customers",
    3: "Low-Engagement Customers"
}

df["Segment_Name"] = df["Cluster"].map(cluster_names)


# =========================================================
# CLUSTER SUMMARY
# =========================================================

cluster_summary = (
    df.groupby(["Cluster", "Segment_Name"])
    .agg(
        Customer_Count=("id", "count"),
        Age=("age", "mean"),
        Income=("income", "mean"),
        Spending_Score=("spending_score", "mean"),
        Purchase_Frequency=("purchase_frequency", "mean"),
        Last_Purchase_Amount=("last_purchase_amount", "mean")
    )
    .reset_index()
)

for col in [
    "Age",
    "Income",
    "Spending_Score",
    "Purchase_Frequency",
    "Last_Purchase_Amount"
]:
    cluster_summary[col] = cluster_summary[col].round(2)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-header">

<h1>🛍️ Customer Segmentation</h1>

<p>
K-Means Clustering Based Customer Analytics Dashboard
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# NAVIGATION
# =========================================================

page = st.selectbox(
    "Select Page",
    [
        "🏠 Dashboard",
        "👥 Customer Segments",
        "📊 Analytics",
        "🔍 Analyze Customer",
        "ℹ️ About Project"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">📊 Project Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            len(df)
        )

    with col2:
        st.metric(
            "Total Features",
            9
        )

    with col3:
        st.metric(
            "Customer Segments",
            4
        )

    with col4:
        st.metric(
            "Algorithm",
            "K-Means"
        )


    st.markdown(
        '<div class="section-title">👥 Customer Distribution</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    data = cluster_summary.sort_values("Cluster")

    ax.bar(
        data["Segment_Name"],
        data["Customer_Count"]
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Distribution by Segment")

    plt.xticks(rotation=15)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


    st.markdown(
        '<div class="section-title">📋 Segment Summary</div>',
        unsafe_allow_html=True
    )

    display_summary = cluster_summary.copy()

    display_summary.columns = [
        "Cluster",
        "Segment Name",
        "Customer Count",
        "Age",
        "Income",
        "Spending Score",
        "Purchase Frequency",
        "Last Purchase Amount"
    ]

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# CUSTOMER SEGMENTS
# =========================================================

elif page == "👥 Customer Segments":

    st.markdown(
        '<div class="section-title">👥 Customer Segments</div>',
        unsafe_allow_html=True
    )

    segment = st.selectbox(
        "Select Customer Segment",
        list(cluster_names.values())
    )

    row = cluster_summary[
        cluster_summary["Segment_Name"] == segment
    ].iloc[0]


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Customer Count",
            int(row["Customer_Count"])
        )

    with col2:
        st.metric(
            "Average Age",
            row["Age"]
        )

    with col3:
        st.metric(
            "Average Income",
            f"${row['Income']:,.2f}"
        )


    st.markdown("### 📈 Segment Characteristics")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Average Spending Score:** "
            f"{row['Spending_Score']}"
        )

        st.write(
            f"**Average Purchase Frequency:** "
            f"{row['Purchase_Frequency']}"
        )

    with col2:

        st.write(
            f"**Average Last Purchase Amount:** "
            f"${row['Last_Purchase_Amount']:,.2f}"
        )


    st.markdown("---")


    if segment == "Premium Customers":

        st.markdown("""
### 💎 Business Insight

High-income and high-spending customers.

**Marketing Focus**

Premium offers, personalized services and loyalty programs.
""")


    elif segment == "Value Spenders":

        st.markdown("""
### 💰 Business Insight

Customers showing relatively high spending behavior.

**Marketing Focus**

Promotional offers and personalized product recommendations.
""")


    elif segment == "Budget Frequent Buyers":

        st.markdown("""
### 🛍️ Business Insight

Customers who purchase frequently but have comparatively lower spending.

**Marketing Focus**

Bundle offers, loyalty rewards and value-based promotions.
""")


    else:

        st.markdown("""
### 📉 Business Insight

Customers showing lower spending and lower purchase engagement.

**Marketing Focus**

Re-engagement campaigns and special offers.
""")


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="section-title">📈 Customer Analytics</div>',
        unsafe_allow_html=True
    )

    analysis = st.selectbox(
        "Select Analysis",
        [
            "Income vs Spending Score",
            "Average Income by Segment",
            "Average Spending Score by Segment",
            "Average Purchase Frequency by Segment"
        ]
    )


    # -----------------------------------------------------
    # INCOME VS SPENDING
    # -----------------------------------------------------

    if analysis == "Income vs Spending Score":

        fig, ax = plt.subplots(figsize=(10, 5.5))

        for cluster in sorted(df["Cluster"].unique()):

            data = df[df["Cluster"] == cluster]

            ax.scatter(
                data["income"],
                data["spending_score"],
                label=cluster_names[cluster],
                alpha=0.65,
                s=45
            )

        ax.set_title("Income vs Spending Score")
        ax.set_xlabel("Annual Income")
        ax.set_ylabel("Spending Score")

        ax.legend(
            loc="best",
            fontsize=9
        )

        ax.grid(alpha=0.25)

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)


    # -----------------------------------------------------
    # AVERAGE INCOME
    # -----------------------------------------------------

    elif analysis == "Average Income by Segment":

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            cluster_summary["Segment_Name"],
            cluster_summary["Income"]
        )

        ax.set_title("Average Income by Customer Segment")
        ax.set_xlabel("Customer Segment")
        ax.set_ylabel("Average Income")

        plt.xticks(rotation=15)
        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)


    # -----------------------------------------------------
    # AVERAGE SPENDING
    # -----------------------------------------------------

    elif analysis == "Average Spending Score by Segment":

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            cluster_summary["Segment_Name"],
            cluster_summary["Spending_Score"]
        )

        ax.set_title(
            "Average Spending Score by Customer Segment"
        )

        ax.set_xlabel("Customer Segment")
        ax.set_ylabel("Average Spending Score")

        plt.xticks(rotation=15)
        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)


    # -----------------------------------------------------
    # PURCHASE FREQUENCY
    # -----------------------------------------------------

    elif analysis == "Average Purchase Frequency by Segment":

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            cluster_summary["Segment_Name"],
            cluster_summary["Purchase_Frequency"]
        )

        ax.set_title(
            "Average Purchase Frequency by Customer Segment"
        )

        ax.set_xlabel("Customer Segment")
        ax.set_ylabel("Average Purchase Frequency")

        plt.xticks(rotation=15)
        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)


# =========================================================
# ANALYZE CUSTOMER
# =========================================================

elif page == "🔍 Analyze Customer":

    st.markdown(
        '<div class="section-title">🔍 Customer Segment Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter customer information to predict the customer segment "
        "using the trained K-Means model."
    )


    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=40
        )

        spending_score = st.number_input(
            "Spending Score",
            min_value=1,
            max_value=100,
            value=70
        )


    with col2:

        income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=60000.0
        )

        purchase_frequency = st.number_input(
            "Purchase Frequency",
            min_value=0,
            value=25
        )


    if st.button(
        "🔍 Analyze Customer",
        use_container_width=True
    ):

        customer = pd.DataFrame({
            "age": [age],
            "income": [income],
            "spending_score": [spending_score],
            "purchase_frequency": [purchase_frequency]
        })


        customer_scaled = scaler.transform(customer)


        predicted_cluster = int(
            kmeans_model.predict(customer_scaled)[0]
        )


        segment = cluster_names[predicted_cluster]


        row = cluster_summary[
            cluster_summary["Cluster"] == predicted_cluster
        ].iloc[0]


        st.success(
            f"Predicted Segment: {segment}"
        )


        st.markdown(
            f"## 🎯 {segment}"
        )

        st.write(
            f"**Predicted Cluster:** Cluster {predicted_cluster}"
        )


        result = pd.DataFrame({
            "Feature": [
                "Age",
                "Income",
                "Spending Score",
                "Purchase Frequency",
                "Last Purchase Amount"
            ],
            "Cluster Average": [
                row["Age"],
                f"${row['Income']:,.2f}",
                row["Spending_Score"],
                row["Purchase_Frequency"],
                f"${row['Last_Purchase_Amount']:,.2f}"
            ]
        })


        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="section-title">ℹ️ About Project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
## Customer Segmentation using K-Means Clustering

### Objective

The main objective of this project is to group customers
based on similar characteristics and purchasing behavior.

### Problem Statement

Businesses often have a large number of customers and
cannot treat every customer in the same way.

Customer segmentation helps identify groups of customers
with similar characteristics.

### Proposed Solution

The project uses the **K-Means Clustering Algorithm**
to divide customers into meaningful segments.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

### Dataset Features

- Customer ID
- Age
- Gender
- Annual Income
- Spending Score
- Membership Years
- Purchase Frequency
- Preferred Category
- Last Purchase Amount

### Methodology

1. Data Collection
2. Data Preprocessing
3. Feature Selection
4. Finding Optimal Clusters
5. K-Means Clustering
6. Model Training
7. Visualization
8. Cluster Analysis
9. Interpretation

### Expected Outcome

Customers are grouped into meaningful segments
that can help businesses improve targeted marketing,
customer understanding and sales strategies.

### Conclusion

K-Means clustering provides an efficient approach
for customer segmentation and data-driven business decisions.
""")
