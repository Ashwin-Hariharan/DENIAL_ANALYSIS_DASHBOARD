# DENIAL_ANALYSIS_DASHBOARD

A comprehensive end-to-end solution to help medical billing analysts and Revenue Cycle Management (RCM) professionals identify, predict, and reduce claim denials using machine learning and interactive visual reporting.

---

**##  Data Input Format**

The Excel file must contain the following columns:

| Column Name         | Description                          |
|---------------------|--------------------------------------|
| CPT Code            | Procedure code for the billed service |
| Insurance Company   | Name of the payer                    |
| Physician Name      | The rendering provider               |
| Payment Amount      | Paid amount by payer (in $)          |
| Balance             | Unpaid amount (in $)                 |
| Denial Reason       | Optional, reason for denial if known |

---

**##  System Capabilities**

### 1.  Identify Top Denied CPT Codes
- Rank CPT codes based on number of denials  
- Calculate denial rate  
- Visualize top CPTs denied

### 2.  Break Down by Payer and Provider
- Show which **insurance companies** and **physicians** receive the most denials

### 3.  Root Cause Analysis
Investigate denial reasons such as:
- Missing modifiers  
- Fee schedule mismatches  
- Non-covered services  
- Documentation or credentialing issues

### 4.  Recommendations
- Improve modifier usage & documentation
- Suggest payer-specific appeal strategies
- Optimize front-desk & billing workflows

### 5.  Machine Learning Predictions
- **Binary Classification:** Will the claim be denied?
- **Multiclass Classification:** What is the denial reason?

### 6.  Visual Reporting
Bar charts displayed in **Streamlit**:
- Denials by CPT
- Denials by Insurance
- Denials by Physician
- Lost Revenue by CPT
- Denial Rate (%)

---

**##  Final Output (Streamlit App)**

### Upload Excel File  
- Get cleaned & enriched data  
- See ML predictions  
- View graphical dashboards  
- Append data to existing file  
- Export summary or prediction results

---

**##  Project Workflow**

###  Step 1: Dataset Upscaling via SMOTE-NC
- Generate ~500 balanced records using SMOTE-NC  
- Handles categorical and numeric fields together

###  Step 2: Streamlit Web App
A dashboard to:
- Upload → Analyze → Predict → Recommend
-
###  Step 3: Excel Data Upload and Merge
- User uploads Excel file  
- File is cleaned and validated  
- Appended to a historical master dataset



###  Step 4: Streamlit Interface

| Button | Function |
|--------|----------|
| Add Additional Data | Upload and merge new Excel data |
| Run Pipeline and Show Results | Launch ML + analytics |
| Start Over | Reset the process |

---

##  Step-by-Step Breakdown

###  Add Additional Data
- Upload Excel file  
- Clean currency, drop empty rows  
- Normalize column names  
- Append to existing master dataset

###  Run Pipeline and Show Results

####  Tab 1: ML Model
- Binary classifier for **denied or not**
- Multiclass classifier for **denial reason**
- Shows:
  - Accuracy
  - Classification Report
  - Confusion Matrix

####  Tab 2: Analysis
- Charts:
  - Denials by CPT, Payer, Provider
  - Lost Revenue
  - Denial Rates
- Root Cause & Recommendations:
  - Displays common issues
  - Provides suggestions per issue

###  Start Over
- Reverts back to Step 4 to upload and rerun pipeline

---


**###  Key Strengths**
- Realistic upsampling using SMOTE-NC  
- Clean, scalable dataset merging  
- No-code interface via Streamlit  
- ML + business insights in one place  
- Root cause suggestions based on trends


---

**##  How to Run Locally**

```bash
git clone https://github.com/yourusername/denial-analysis-dashboard.git
cd denial-analysis-dashboard
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run Denial_Analysis_Application.py
