import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("customer_segmentation_data.csv")


# =========================================================
# 2. FEATURES FOR K-MEANS
# =========================================================

features = [
    "age",
    "income",
    "spending_score",
    "purchase_frequency"
]

X = df[features]


# =========================================================
# 3. DATA SCALING
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================================================
# 4. K-MEANS MODEL
# =========================================================

optimal_k = 4

kmeans_model = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

kmeans_model.fit(X_scaled)


# =========================================================
# 5. CLUSTER LABELS
# =========================================================

df["Cluster"] = kmeans_model.labels_


# =========================================================
# 6. CLUSTER NAMES
# =========================================================

cluster_names = {
    0: "Budget Frequent Buyers",
    1: "Value Spenders",
    2: "Premium Customers",
    3: "Low-Engagement Customers"
}

df["Segment_Name"] = df["Cluster"].map(cluster_names)


# =========================================================
# 7. CLUSTER SUMMARY
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
# 8. CUSTOMER DISTRIBUTION CHART
# =========================================================

def distribution_chart():

    fig, ax = plt.subplots(figsize=(8, 4.5))

    data = cluster_summary.sort_values("Cluster")

    ax.bar(
        data["Segment_Name"],
        data["Customer_Count"]
    )

    ax.set_title(
        "Customer Distribution by Segment",
        fontsize=15,
        pad=12
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")

    ax.tick_params(
        axis="x",
        rotation=15
    )

    plt.tight_layout()

    return fig


# =========================================================
# 9. INCOME VS SPENDING SCORE
# =========================================================

def scatter_chart():

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for cluster in sorted(df["Cluster"].unique()):

        data = df[df["Cluster"] == cluster]

        ax.scatter(
            data["income"],
            data["spending_score"],
            label=cluster_names[cluster],
            alpha=0.65,
            s=35
        )

    ax.set_title(
        "Income vs Spending Score",
        fontsize=15,
        pad=12
    )

    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")

    ax.legend(
        loc="best",
        fontsize=8
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    return fig


# =========================================================
# 10. AVERAGE INCOME
# =========================================================

def income_chart():

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(
        cluster_summary["Segment_Name"],
        cluster_summary["Income"]
    )

    ax.set_title(
        "Average Income by Customer Segment",
        fontsize=15,
        pad=12
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Average Income")

    ax.tick_params(
        axis="x",
        rotation=15
    )

    plt.tight_layout()

    return fig


# =========================================================
# 11. AVERAGE SPENDING SCORE
# =========================================================

def spending_chart():

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(
        cluster_summary["Segment_Name"],
        cluster_summary["Spending_Score"]
    )

    ax.set_title(
        "Average Spending Score by Customer Segment",
        fontsize=15,
        pad=12
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Average Spending Score")

    ax.tick_params(
        axis="x",
        rotation=15
    )

    plt.tight_layout()

    return fig


# =========================================================
# 12. AVERAGE PURCHASE FREQUENCY
# =========================================================

def frequency_chart():

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(
        cluster_summary["Segment_Name"],
        cluster_summary["Purchase_Frequency"]
    )

    ax.set_title(
        "Average Purchase Frequency by Customer Segment",
        fontsize=15,
        pad=12
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Average Purchase Frequency")

    ax.tick_params(
        axis="x",
        rotation=15
    )

    plt.tight_layout()

    return fig


# =========================================================
# 13. SEGMENT DETAILS
# =========================================================

def show_segment(segment_name):

    row = cluster_summary[
        cluster_summary["Segment_Name"] == segment_name
    ].iloc[0]

    details = f"""
## {segment_name}

**Customer Count:** {int(row["Customer_Count"])}

**Average Age:** {row["Age"]}

**Average Income:** {row["Income"]:,.2f}

**Average Spending Score:** {row["Spending_Score"]}

**Average Purchase Frequency:** {row["Purchase_Frequency"]}

**Average Last Purchase Amount:** {row["Last_Purchase_Amount"]:,.2f}
"""

    if segment_name == "Premium Customers":

        insight = """
## 💎 Business Insight

High-income and high-spending customers.

### Marketing Focus

Premium offers, personalized services and loyalty programs.
"""

    elif segment_name == "Value Spenders":

        insight = """
## 💰 Business Insight

Customers showing relatively high spending behavior.

### Marketing Focus

Promotional offers and personalized product recommendations.
"""

    elif segment_name == "Budget Frequent Buyers":

        insight = """
## 🛍️ Business Insight

Customers who purchase frequently but have comparatively lower spending.

### Marketing Focus

Bundle offers, loyalty rewards and value-based promotions.
"""

    else:

        insight = """
## 📉 Business Insight

Customers showing lower spending and lower purchase engagement.

### Marketing Focus

Re-engagement campaigns and special offers.
"""

    return details, insight


# =========================================================
# 14. ANALYTICS DROPDOWN
# =========================================================

def update_analytics(choice):

    if choice == "Income vs Spending Score":
        return scatter_chart()

    elif choice == "Average Income by Segment":
        return income_chart()

    elif choice == "Average Spending Score by Segment":
        return spending_chart()

    elif choice == "Average Purchase Frequency by Segment":
        return frequency_chart()

    return scatter_chart()


# =========================================================
# 15. CUSTOMER PREDICTION
# =========================================================

def analyze_customer(
    age,
    income,
    spending_score,
    purchase_frequency
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

    result = f"""
# 🎯 {segment}

### Predicted Cluster

**Cluster {predicted_cluster}**

---

### Cluster Characteristics

| Feature | Average |
|---|---:|
| Age | {row["Age"]} |
| Income | {row["Income"]:,.2f} |
| Spending Score | {row["Spending_Score"]} |
| Purchase Frequency | {row["Purchase_Frequency"]} |
| Last Purchase Amount | {row["Last_Purchase_Amount"]:,.2f} |
"""

    return result


# =========================================================
# 16. CUSTOM CSS
# =========================================================

custom_css = """

/* =========================================
   MAIN PAGE
========================================= */

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding: 18px 25px 30px 25px !important;
    background-color: #eef2f7 !important;
}


/* =========================================
   REMOVE GRADIO FOOTER
========================================= */

footer,
.gradio-container footer,
[class*="footer"] {
    display: none !important;
}


/* =========================================
   HEADER
========================================= */

#main-header {
    text-align: center;
    padding: 28px 20px;
    margin-bottom: 18px;
    border-radius: 16px;
    background: linear-gradient(
        135deg,
        #2c4a6e,
        #4a7ab5
    ) !important;
}

#main-header h1 {
    font-size: 32px;
    margin: 0 0 7px 0;
    color: white !important;
}

#main-header p {
    font-size: 15px;
    margin: 0;
    color: white !important;
}


/* =========================================
   TABS
========================================= */

.tab-nav {
    display: flex !important;
    gap: 6px !important;
    margin-bottom: 15px !important;
}

.tab-nav button {
    flex: 1 !important;
    min-height: 42px !important;
    padding: 9px 15px !important;
    font-size: 14px !important;
    justify-content: center !important;
}


/* =========================================
   TAB CONTENT
========================================= */

.tabitem {
    padding: 8px 5px 25px 5px !important;
}


/* =========================================
   HEADINGS
========================================= */

h2 {
    margin-top: 8px !important;
    margin-bottom: 14px !important;
}

h3 {
    margin-top: 8px !important;
    margin-bottom: 12px !important;
}


/* =========================================
   ROWS
========================================= */

.gradio-row {
    align-items: stretch !important;
    gap: 15px !important;
}


/* =========================================
   PLOTS
========================================= */

.plot-container {
    width: 100% !important;
    min-height: 430px !important;
}


/* =========================================
   DATAFRAME
========================================= */

.dataframe {
    width: 100% !important;
}


/* =========================================
   INPUTS
========================================= */

input,
textarea,
button {
    border-radius: 9px !important;
}


/* =========================================
   BUTTON
========================================= */

button {
    min-height: 42px !important;
}


/* =========================================
   MARKDOWN
========================================= */

.markdown {
    line-height: 1.55 !important;
}


/* =========================================
   NUMBER / TEXTBOX CARDS
========================================= */

.gradio-number,
.gradio-textbox {
    border-radius: 10px !important;
}


/* =========================================
   ANALYTICS PLOT AREA
========================================= */

#analytics-area {
    max-width: 950px !important;
    margin: auto !important;
}


/* =========================================
   SEGMENT AREA
========================================= */

#segment-area {
    max-width: 1000px !important;
    margin: auto !important;
}


/* =========================================
   PREDICTION AREA
========================================= */

#prediction-area {
    max-width: 850px !important;
    margin: auto !important;
}

"""


# =========================================================
# 17. MAIN UI
# =========================================================

with gr.Blocks(
    title="Customer Segmentation Dashboard",
    css=custom_css
) as demo:

    # =====================================================
    # HEADER
    # =====================================================

    gr.HTML(
        """
        <div id="main-header">
            <h1>🛍️ Customer Segmentation</h1>
            <p>
                K-Means Clustering Based Customer Analytics Dashboard
            </p>
        </div>
        """
    )


    # =====================================================
    # TAB 1 — DASHBOARD
    # =====================================================

    with gr.Tab("🏠 Dashboard"):

        gr.Markdown("## 📊 Project Overview")

        with gr.Row(equal_height=True):

            with gr.Column():
                gr.Number(
                    value=len(df),
                    label="Total Customers",
                    interactive=False
                )

            with gr.Column():
                gr.Number(
                    value=9,
                    label="Total Features",
                    interactive=False
                )

            with gr.Column():
                gr.Number(
                    value=4,
                    label="Customer Segments",
                    interactive=False
                )

            with gr.Column():
                gr.Textbox(
                    value="K-Means",
                    label="Algorithm",
                    interactive=False
                )


        gr.Markdown("## 👥 Customer Distribution")

        gr.Plot(
            value=distribution_chart()
        )


        gr.Markdown("## 📋 Segment Summary")

        gr.Dataframe(
            value=cluster_summary,
            interactive=False,
            wrap=True
        )


    # =====================================================
    # TAB 2 — SEGMENTS
    # =====================================================

    with gr.Tab("👥 Segments"):

        with gr.Column(elem_id="segment-area"):

            gr.Markdown("## 👥 Customer Segments")

            segment_dropdown = gr.Dropdown(
                choices=list(cluster_names.values()),
                value="Premium Customers",
                label="Select Customer Segment"
            )

            with gr.Row(equal_height=True):

                with gr.Column():
                    segment_details = gr.Markdown()

                with gr.Column():
                    segment_insight = gr.Markdown()


            segment_dropdown.change(
                fn=show_segment,
                inputs=segment_dropdown,
                outputs=[
                    segment_details,
                    segment_insight
                ]
            )

            demo.load(
                fn=show_segment,
                inputs=segment_dropdown,
                outputs=[
                    segment_details,
                    segment_insight
                ]
            )


    # =====================================================
    # TAB 3 — ANALYTICS
    # =====================================================

    with gr.Tab("📊 Analytics"):

        with gr.Column(elem_id="analytics-area"):

            gr.Markdown("## 📈 Customer Analytics")

            analytics_dropdown = gr.Dropdown(
                choices=[
                    "Income vs Spending Score",
                    "Average Income by Segment",
                    "Average Spending Score by Segment",
                    "Average Purchase Frequency by Segment"
                ],
                value="Income vs Spending Score",
                label="Select Analysis"
            )

            analytics_plot = gr.Plot(
                value=scatter_chart()
            )


            analytics_dropdown.change(
                fn=update_analytics,
                inputs=analytics_dropdown,
                outputs=analytics_plot
            )


    # =====================================================
    # TAB 4 — ANALYZE CUSTOMER
    # =====================================================

    with gr.Tab("🔍 Analyze Customer"):

        with gr.Column(elem_id="prediction-area"):

            gr.Markdown("## 🔍 Customer Segment Prediction")

            gr.Markdown(
                """
                Enter the customer information below.
                The trained **K-Means model** will predict
                the customer's segment.
                """
            )


            with gr.Row(equal_height=True):

                with gr.Column():
                    age_input = gr.Number(
                        label="Age",
                        value=40
                    )

                with gr.Column():
                    income_input = gr.Number(
                        label="Annual Income",
                        value=60000
                    )


            with gr.Row(equal_height=True):

                with gr.Column():
                    spending_input = gr.Number(
                        label="Spending Score",
                        value=70
                    )

                with gr.Column():
                    frequency_input = gr.Number(
                        label="Purchase Frequency",
                        value=25
                    )


            analyze_button = gr.Button(
                "🔍 Analyze Customer",
                variant="primary"
            )


            prediction_output = gr.Markdown()


            analyze_button.click(
                fn=analyze_customer,
                inputs=[
                    age_input,
                    income_input,
                    spending_input,
                    frequency_input
                ],
                outputs=prediction_output
            )


# =========================================================
# 18. LAUNCH
# =========================================================

if __name__ == "__main__":
    demo.launch()
