import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib


st.set_page_config(page_title="Denial Analysis", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1; font-size: 50px;'>Denial Analysis Dashboard</h1>
    <hr style='border: 2px solid #2E86C1;'>
    """,
    unsafe_allow_html=True
)


if 'add_data_clicked' not in st.session_state:
    st.session_state.add_data_clicked = False
if st.button("Add additional data"):
    st.session_state.add_data_clicked = True


def save_barplot(data, x, y, title, xlabel, ylabel, palette):
    fig, ax = plt.subplots(figsize=(6, 4))  
    data[y] = data[y].astype(str)
    top_data = data.head(10)
    sns.barplot(x=x, y=y, data=top_data, palette=palette, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=8, padding=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig

def classify(df):
    

    categorical_cols = ['cpt_code', 'insurance_company', 'physician_name']
    for col in categorical_cols:
        encoder = joblib.load(f"{col}_encoder.pkl")  # Load the encoder
        df[col + '_enc'] = encoder.transform(df[col].astype(str))  # Transform using encoder

    # Prepare features
    features = ['cpt_code_enc', 'insurance_company_enc', 'physician_name_enc', 'payment_amount', 'balance']
    X = df[features]

    # Load pre-trained model
    model = joblib.load("denial_model.pkl")

    if st.button("Run Model"):
        st.write("=== Prediction on Entire Dataset ===")
        df['Predicted_Denied'] = model.predict(X)
        df['Predicted_Denied'] = df['Predicted_Denied'].map({1: "Denied", 0: "Not Denied"})

        display_cols = ['cpt_code', 'insurance_company', 'physician_name', 'payment_amount', 'balance', 'Predicted_Denied', 'denial_reason']
        st.write("###  Prediction Results:")
        st.dataframe(df[display_cols].head(50))

def tabs():
    st.markdown("""
    <style>
    div[data-baseweb="tab"] {
        font-size: 18px;
        color: #2E86C1;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        border: 1px solid #2E86C1;
        margin-right: 10px;
    }
    div[data-baseweb="tab"]:hover {
        background-color: #e8f4fa;
        color: #1B4F72;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #2E86C1;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["ML Prediction", "Data Analysis"])

    with tab1:
        df = pd.read_excel(r'C:\Users\ashwi\GUVI_Projects\Job\Tensaw\Ass\AR_performance_review_synthetic.xlsx')

        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df['denial_reason'] = df['denial_reason'].fillna("Not Denied")
        df['denied'] = (df['denial_reason'] != "Not Denied").astype(int)

        label_encoders = {}
        categorical_cols = ['cpt_code', 'insurance_company', 'physician_name']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col])  # Store encoded values in new columns
            label_encoders[col] = le

        df['payment_amount'] = pd.to_numeric(df['payment_amount'], errors='coerce').fillna(0)
        df['balance'] = pd.to_numeric(df['balance'], errors='coerce').fillna(0)

        features = ['cpt_code_enc', 'insurance_company_enc', 'physician_name_enc', 'payment_amount', 'balance']
        X = df[features]
        y_binary = df['denied']
        y_multiclass = df[df['denied'] == 1]['denial_reason']
        X_multiclass = df[df['denied'] == 1][features]

        X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_binary, test_size=0.2, random_state=42)
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multiclass, y_multiclass, test_size=0.2, random_state=42)

        clf_bin = RandomForestClassifier(random_state=42, class_weight="balanced")
        clf_bin.fit(X_train_bin, y_train_bin)
        y_pred_bin = clf_bin.predict(X_test_bin)

        st.write("=== Denied or Not  ===")
        accuracy = accuracy_score(y_test_bin, y_pred_bin)
        report = classification_report(y_test_bin, y_pred_bin, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.text("Classification Report:")
        st.table(report_df)

        le_reason = LabelEncoder()
        y_train_multi_enc = le_reason.fit_transform(y_train_multi)
        y_test_multi_enc = le_reason.transform(y_test_multi)

        clf_multi = RandomForestClassifier(random_state=42, class_weight="balanced")
        clf_multi.fit(X_train_multi, y_train_multi_enc)
        y_pred_multi = clf_multi.predict(X_test_multi)

        st.write("=== Denial Reason (Multiclass) ===")
        accuracy = accuracy_score(y_test_multi_enc, y_pred_multi)
        report = classification_report(y_test_multi_enc, y_pred_multi, target_names=le_reason.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write("Classification Report:")
        st.table(report_df)

    with tab2:
        st.markdown("""
        <style>
        div[data-baseweb="tab"] {
            font-size: 18px;
            color: #2E86C1;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
            border: 1px solid #2E86C1;
            margin-right: 10px;
        }
        div[data-baseweb="tab"]:hover {
            background-color: #e8f4fa;
            color: #1B4F72;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background-color: #2E86C1;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

        tab_a, tab_b = st.tabs(["Analysis Part", "Solution and Root Cause"])
        with tab_a:
            df['denied'] = df['denial_reason'].notna()

            denials_by_cpt = df[df['denied']].groupby('cpt_code').size().reset_index(name='denial_count')
            total_count = len(df)
            total_by_cpt = df.groupby('cpt_code').size().reset_index(name='total_count')

            merged = pd.merge(denials_by_cpt, total_by_cpt, on='cpt_code')
            merged['denial_rate'] = (merged['denial_count'] / total_count) * 100
            merged_filtered = merged[(merged['denial_rate'] > 0) & (merged['denial_rate'] < 100)]

            denials_by_payer = df[df['denied']].groupby('insurance_company').size().sort_values(ascending=False).reset_index(name='denial_count')
            denials_by_provider = df[df['denied']].groupby('physician_name').size().sort_values(ascending=False).reset_index(name='denial_count')
            lost_revenue = df[df['denied']].groupby('cpt_code')['balance'].sum().sort_values(ascending=False).reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                fig1 = save_barplot(
                    denials_by_cpt.sort_values('denial_count', ascending=False),
                    'denial_count', 'cpt_code',
                    "Top Denied CPT Codes", "Denial Count", "CPT Code", 'Reds_r'
                )
                st.pyplot(fig1)
            with col2:
                fig2 = save_barplot(
                    merged_filtered.sort_values('denial_rate', ascending=False),
                    'denial_rate', 'cpt_code',
                    "Top CPTs by Denial Rate (%)", "Denial Rate (%)", "CPT Code", 'Greens_r'
                )
                st.pyplot(fig2)
            col3, col4 = st.columns(2)
            with col3:
                fig3 = save_barplot(
                    denials_by_payer,
                    'denial_count', 'insurance_company',
                    "Top Payers with Most Denials", "Denial Count", "Insurance Company", 'Purples_r'
                )
                st.pyplot(fig3)
            with col4:
                fig4 = save_barplot(
                    denials_by_provider,
                    'denial_count', 'physician_name',
                    "Top Providers with Most Denials", "Denial Count", "Physician", 'Oranges_r'
                )
                st.pyplot(fig4)
            fig5 = save_barplot(
                lost_revenue,
                'balance', 'cpt_code',
                "Top CPTs by Lost Revenue", "Total Unpaid Balance ($)", "CPT Code", 'Blues_r'
            )
            st.pyplot(fig5)


        with tab_b:
            section_issue = df['denial_reason'].dropna().astype(str).unique().tolist()

            section_solution = """
                ####  Recommendations:

                **1. Missing information**
                -  Use complete and correct details.
                -  Train staff on payer-specific documentation.

                **2. Charge exceeds fee schedule**
                -  Compare charges with payer schedules regularly.
                -  Appeal denials with proper justification.

                **3. Non-covered service**
                -  Check plan coverage before providing care.
                -  Inform patients of potential out-of-pocket costs.
                """

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Root Cause of Denial")
                filtered_reasons = [r for r in section_issue if r != 'Not Denied']
                for reason in filtered_reasons:

                    if ' - ' in reason:
                        code, desc = reason.split(" - ", 1)
                    else:
                        code, desc = reason, ''
                    
                    st.markdown(f"""
                    <div style='color:black; padding:4px; font-size:16px'>
                        <b>{code}</b>: {desc}
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("### Recommendations for Denials")
                st.markdown(section_solution)

if st.session_state.add_data_clicked == True:

    if st.session_state.add_data_clicked:
        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    def load_excel_with_dynamic_header(file):
        raw = pd.read_excel(file, header=None)
        for i, row in raw.iterrows():
            if row.notna().sum() >= 3:
                header_row_index = i
                break
        return pd.read_excel(file, skiprows=header_row_index)

    if uploaded_file:
        try:
            df = load_excel_with_dynamic_header(uploaded_file)

            expected_cols = ['#', 'CPT Code', 'Insurance Company', 'Physician Name', 'Payment Amount', 'Balance', 'Denial Reason']
            if list(df.columns)[:len(expected_cols)] != expected_cols:
                st.error("Excel format mismatch. Expected columns: " + ", ".join(expected_cols))
            else:
                df.drop(columns=['#'], inplace=True)
                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
                df['payment_amount'] = df['payment_amount'].replace('[\$,]', '', regex=True).astype(float)
                df['balance'] = df['balance'].replace('[\$,]', '', regex=True).astype(float)
                df['payment_amount'].fillna(df['payment_amount'].median(), inplace=True)
                df['balance'].fillna(df['balance'].median(), inplace=True)
                df['denial_reason'].fillna("Not Denied", inplace=True)

                existing_path = r'C:\Users\ashwi\GUVI_Projects\Job\Tensaw\Ass\AR_performance_review_synthetic.xlsx'
                df_exist = pd.read_excel(existing_path)
                combined_df = pd.concat([df_exist, df], ignore_index=True)
                combined_df.to_excel(existing_path, index=False)
                classify(df)
                st.success(" Data successfully appended!")

        except Exception as e:
            st.error(f"Error loading file: {e}")

if st.button("Run pipeline and show results"):
    tabs()

if st.button("Start Over"):
    st.session_state.add_data_clicked = False 
    st.rerun()
