import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_excel(r'C:\Users\ashwi\GUVI_Projects\Job\Tensaw\Ass\AR_performance_review_synthetic.xlsx')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df['denial_reason'] = df['denial_reason'].fillna("Not Denied")
df['denied'] = (df['denial_reason'] != "Not Denied").astype(int)

categorical_cols = ['cpt_code', 'insurance_company', 'physician_name']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    label_encoders[col] = le

df['payment_amount'] = pd.to_numeric(df['payment_amount'], errors='coerce').fillna(0)
df['balance'] = pd.to_numeric(df['balance'], errors='coerce').fillna(0)

features = ['cpt_code_enc', 'insurance_company_enc', 'physician_name_enc', 'payment_amount', 'balance']
X = df[features]
y = df['denied']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(df.head(5))
print(y_pred)

joblib.dump(clf, 'denial_model.pkl')
for col in categorical_cols:
    joblib.dump(label_encoders[col], f'{col}_encoder.pkl')

print("\nModel and encoders saved successfully.")

