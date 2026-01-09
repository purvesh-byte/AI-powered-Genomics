import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
import logging
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import spacy

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Try to load spaCy model, fallback to basic NLP if not available
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except OSError:
    logging.warning("spaCy model not found. Using basic NLP features.")
    SPACY_AVAILABLE = False
    nlp = None

class ClinicalTextAnalyzer:
    def __init__(self):
        self.symptom_keywords = [
            'fever', 'pain', 'fatigue', 'weakness', 'seizure', 'tremor',
            'headache', 'nausea', 'vomiting', 'dizziness', 'swelling',
            'rash', 'paleness', 'jaundice', 'breathing', 'cardiac',
            'neurological', 'developmental', 'growth', 'cognitive'
        ]
        self.family_terms = ['mother', 'father', 'parent', 'sibling', 'brother', 
                           'sister', 'family', 'grandparent', 'aunt', 'uncle']
    
    def extract_symptoms(self, text):
        """Extract symptoms from clinical notes using keyword matching"""
        if not text or pd.isna(text):
            return []
        
        text_lower = str(text).lower()
        found_symptoms = []
        
        for symptom in self.symptom_keywords:
            if symptom in text_lower:
                found_symptoms.append(symptom)
        
        return found_symptoms
    
    def extract_family_history(self, text):
        """Extract family medical history mentions"""
        if not text or pd.isna(text):
            return []
        
        text_lower = str(text).lower()
        sentences = re.split(r'[.!?]+', text_lower)
        family_sentences = []
        
        for sentence in sentences:
            if any(term in sentence for term in self.family_terms):
                family_sentences.append(sentence.strip())
        
        return family_sentences
    
    def calculate_symptom_density(self, text):
        """Calculate symptom mentions per sentence"""
        if not text or pd.isna(text):
            return 0
        
        sentences = re.split(r'[.!?]+', str(text))
        symptom_count = len(self.extract_symptoms(text))
        return symptom_count / len(sentences) if sentences else 0

# Initialize NLP analyzer
clinical_analyzer = ClinicalTextAnalyzer()

# ===============================
# Feature Setup (dropdowns + names)
# ===============================
FEATURES = {
    "Patient Age": "Patient Age",
    "Genes in mother": "Genes in mother's side",
    "Inherited from father": "Inherited from father",
    "Maternal gene": "Maternal gene",
    "Paternal gene": "Paternal gene",
    "Status": "Status",
    "Respiratory Rate": "Respiratory Rate (breaths/min)",
    "Heart Rate": "Heart Rate (rates/min",
    "Follow-up": "Follow-up",
    "Gender": "Gender",
    "Birth asphyxia": "Birth asphyxia",
    "Autopsy birth defect": "Autopsy shows birth defect (if applicable)",
    "Folic acid": "Folic acid details (peri-conceptional)",
    "H/O serious maternal illness": "H/O serious maternal illness",
    "H/O radiation exposure (x-ray)": "H/O radiation exposure (x-ray)",
    "H/O substance abuse": "H/O substance abuse",
    "Assisted conception": "Assisted conception IVF/ART",
    "History of anomalies in previous pregnancies": "History of anomalies in previous pregnancies",
    "No. of abortion": "No. of previous abortion",
    "Birth defects": "Birth defects",
    "Blood test result": "Blood test result",
    "Symptom 1": "Symptom 1",
    "Symptom 2": "Symptom 2",
    "Symptom 3": "Symptom 3",
    "Symptom 4": "Symptom 4",
    "Symptom 5": "Symptom 5",
    "White Blood Cell Count": "White Blood cell count (thousand per microliter)",
    "Blood Cell Count": "Blood cell count (mcL)"
}

# NLP Features
NLP_FEATURES = {
    "clinical_notes": "Clinical Notes",
    "family_history": "Family History Description",
    "patient_symptoms": "Patient Reported Symptoms"
}

TARGET_COLUMN = "Genetic Disorder"

DROPDOWN_VALUES = {
    "Patient Age": list(range(0, 16)),
    "Genes in mother": [1, 2],
    "Inherited from father": [1, 2],
    "Maternal gene": [1, 2],
    "Paternal gene": [1, 2],
    "Status": [1, 2],
    "Respiratory Rate": [1, 2],
    "Heart Rate": [1, 2],
    "Follow-up": ["High", "Low"],
    "Gender": [1, 2, 3],
    "Birth asphyxia": [1, 2],
    "Autopsy birth defect": [1, 2],
    "Folic acid": [1, 2],
    "H/O serious maternal illness": [1, 2],
    "H/O radiation exposure (x-ray)": [1, 2],
    "H/O substance abuse": [1, 2],
    "Assisted conception": [1, 2],
    "History of anomalies in previous pregnancies": [1, 2],
    "No. of abortion": list(range(0, 7)),
    "Birth defects": ["Singular", "Multiple", "Unknown"],
    "Blood test result": ["normal", "abnormal", "inconclusive", "Unknown"],
    "Symptom 1": [0, 1],
    "Symptom 2": [0, 1],
    "Symptom 3": [0, 1],
    "Symptom 4": [0, 1],
    "Symptom 5": [0, 1],
    "White Blood Cell Count": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    "Blood Cell Count": [4.0, 5.0, 6.0]
}

def extract_nlp_features(df, nlp_columns):
    """Extract NLP-based features from text data"""
    nlp_features = pd.DataFrame(index=df.index)
    
    # Combine all text columns for analysis
    all_text = ""
    for col in nlp_columns:
        if col in df.columns:
            all_text += " " + df[col].fillna('').astype(str)
    
    # Text statistics
    nlp_features['text_length'] = all_text.str.len()
    nlp_features['word_count'] = all_text.apply(lambda x: len(str(x).split()))
    
    # Sentiment analysis
    nlp_features['sentiment'] = all_text.apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    
    # Symptom extraction features
    nlp_features['symptom_count'] = all_text.apply(
        lambda x: len(clinical_analyzer.extract_symptoms(str(x)))
    )
    
    nlp_features['symptom_density'] = all_text.apply(
        lambda x: clinical_analyzer.calculate_symptom_density(str(x))
    )
    
    # Family history mentions
    nlp_features['family_mentions'] = all_text.apply(
        lambda x: len(clinical_analyzer.extract_family_history(str(x)))
    )
    
    # TF-IDF features (limited to avoid dimensionality issues)
    try:
        tfidf = TfidfVectorizer(max_features=20, stop_words='english')
        tfidf_features = tfidf.fit_transform(all_text.fillna(''))
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(), 
            columns=[f"tfidf_{i}" for i in range(tfidf_features.shape[1])],
            index=df.index
        )
        nlp_features = pd.concat([nlp_features, tfidf_df], axis=1)
    except Exception as e:
        logging.warning(f"TF-IDF feature extraction failed: {e}")
    
    return nlp_features

def load_and_preprocess_data():
    """Load and preprocess the dataset with error handling"""
    try:
        df = pd.read_csv("train.csv")
        df = df.dropna(subset=[TARGET_COLUMN])
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Define full feature set
        full_features = list(FEATURES.values())
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(0)
        
        # Numeric conversion
        numeric_cols = [
            "Patient Age", "No. of previous abortion", "Symptom 1", "Symptom 2", "Symptom 3",
            "Symptom 4", "Symptom 5", "White Blood cell count (thousand per microliter)",
            "Blood cell count (mcL)"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        # Feature Engineering
        df["Symptom Score"] = df["Symptom 1"] + df["Symptom 2"] + df["Symptom 3"] + df["Symptom 4"] + df["Symptom 5"]
        
        # Add dummy NLP columns for training if they don't exist
        for nlp_col in NLP_FEATURES.values():
            if nlp_col not in df.columns:
                df[nlp_col] = ""  # Add empty NLP columns for training
        
        return df, full_features
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Load data
df, full_features = load_and_preprocess_data()

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[TARGET_COLUMN])

# Prepare features with NLP
X_structured = df[full_features].copy()
X_structured["Symptom Score"] = df["Symptom Score"]

# Extract NLP features
nlp_feature_df = extract_nlp_features(df, list(NLP_FEATURES.values()))

# Combine structured and NLP features
X = pd.concat([X_structured, nlp_feature_df], axis=1)

# Encode categoricals with TargetEncoder
cat_cols = X_structured.select_dtypes(include="object").columns.tolist()
target_encoder = TargetEncoder()
X[cat_cols] = target_encoder.fit_transform(X[cat_cols], y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ===============================
# Train Models
# ===============================
try:
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
    )
    xgb_model.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    logging.info("Models trained successfully")
    logging.info(f"XGBoost Accuracy: {xgb_acc:.2f}, Random Forest Accuracy: {rf_acc:.2f}")
    
except Exception as e:
    logging.error(f"Error training models: {e}")
    raise

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return render_template(
        "index.html",
        columns=list(FEATURES.keys()),
        nlp_columns=list(NLP_FEATURES.keys()),
        dropdowns=DROPDOWN_VALUES,
        rf_acc=round(rf_acc * 100, 2),
        xgb_acc=round(xgb_acc * 100, 2)
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = {}
        nlp_input = {}
        
        # Validate and collect structured user input
        for label, col_name in FEATURES.items():
            val = request.form.get(label)
            if val is None:
                return jsonify({"error": f"Missing value for {label}"}), 400
                
            # Better type conversion
            try:
                if val.replace('.', '').replace('-', '').isdigit():
                    val = float(val) if '.' in val else int(val)
                # Keep as string if not convertible
            except ValueError:
                pass  # Keep as string
                
            user_input[col_name] = val

        # Collect NLP input
        for nlp_key, nlp_col in NLP_FEATURES.items():
            nlp_input[nlp_col] = request.form.get(nlp_key, "")
        
        # Build input DataFrame
        input_structured = pd.DataFrame([user_input])
        
        # Feature engineering for structured data
        symptom_cols = ["Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4", "Symptom 5"]
        input_structured["Symptom Score"] = sum([input_structured[col].iloc[0] for col in symptom_cols if col in input_structured])
        
        # Combine with NLP data
        input_combined = pd.concat([input_structured, pd.DataFrame([nlp_input])], axis=1)
        
        # Extract NLP features
        nlp_features = extract_nlp_features(input_combined, list(NLP_FEATURES.values()))
        
        # Combine all features
        final_input = pd.concat([input_structured, nlp_features], axis=1)
        
        # Ensure column order matches training
        for col in X.columns:
            if col not in final_input.columns:
                final_input[col] = 0  # Add missing columns with default value
        
        final_input = final_input[X.columns]
        
        # Apply target encoder to categorical columns
        final_input[cat_cols] = target_encoder.transform(final_input[cat_cols])
        
        # Make predictions
        rf_pred_encoded = rf_model.predict(final_input)[0]
        rf_pred = label_encoder.inverse_transform([rf_pred_encoded])[0]

        xgb_pred_encoded = xgb_model.predict(final_input)[0]
        xgb_pred = label_encoder.inverse_transform([xgb_pred_encoded])[0]
        
        # Extract NLP insights for display
        all_text = " ".join([str(nlp_input[col]) for col in NLP_FEATURES.values()])
        nlp_insights = {
            "extracted_symptoms": clinical_analyzer.extract_symptoms(all_text),
            "family_history_mentions": clinical_analyzer.extract_family_history(all_text),
            "sentiment": TextBlob(all_text).sentiment.polarity,
            "symptom_count": len(clinical_analyzer.extract_symptoms(all_text)),
            "word_count": len(all_text.split())
        }

        return jsonify({
            "rf_prediction": rf_pred,
            "xgb_prediction": xgb_pred,
            "rf_accuracy": round(rf_acc * 100, 2),
            "xgb_accuracy": round(xgb_acc * 100, 2),
            "nlp_insights": nlp_insights
        })
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    """Endpoint for real-time text analysis"""
    try:
        text = request.json.get('text', '')
        
        insights = {
            "symptom_count": len(clinical_analyzer.extract_symptoms(text)),
            "family_mentions": len(clinical_analyzer.extract_family_history(text)),
            "sentiment": TextBlob(text).sentiment.polarity,
            "word_count": len(text.split())
        }
        
        return jsonify(insights)
        
    except Exception as e:
        logging.error(f"Text analysis error: {e}")
        return jsonify({"error": "Text analysis failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)