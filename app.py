from flask import Flask,request,render_template
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import base64

app=Flask(__name__)

#Load both models

clf=joblib.load("model/model_classifier.pkl")
reg=joblib.load("model/model_regressor.pkl")

# Load label encoders
try:
    label_encoders = joblib.load("model/label_encoders.pkl")
    print("Label encoders loaded successfully!")
except:
    label_encoders = {}
    print("Warning: Label encoders not found")

# Function to generate SHAP explanations
def generate_shap_plot(model, data):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, data, plot_type='bar', show=False)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return f'data:image/png;base64,{plot_url}'
    except Exception as e:
        print(f"SHAP plot error: {e}")
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    
    #Get input from form
    form_data = request.form.to_dict()
    
    # Map form inputs to feature order (must match training data order)
    feature_names = ['loan_id', 'no_of_dependents', 'education', 'self_employed', 
                     'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
                     'residential_assets_value', 'commercial_assets_value', 
                     'luxury_assets_value', 'bank_asset_value']
    
    # Create data in correct order
    data_list = []
    for feature in feature_names:
        value = float(form_data.get(feature, 0))
        data_list.append(value)
    
    # Convert to numpy array
    data_array = np.array([data_list])
    
    # Create DataFrame with feature names for proper processing
    df_input = pd.DataFrame(data_array, columns=feature_names)
    
    # Apply label encoders to categorical columns
    for col, encoder in label_encoders.items():
        if col in df_input.columns and col != 'loan_status':
            try:
                # Convert value to int then to string for encoding
                val = int(df_input[col].iloc[0])
                # Map numeric values to the correct strings
                # From form: 0=Graduate/No, 1=Not Graduate/Yes
                if col == 'education':
                    val_str = 'Graduate' if val == 0 else 'Not Graduate'
                elif col == 'self_employed':
                    val_str = 'No' if val == 0 else 'Yes'
                df_input[col] = encoder.transform([val_str])[0]
            except Exception as e:
                print(f"Encoding error for {col}: {e}")
    
    # Get final data array
    data_array = df_input.values

    #Loan Approval Prediction

    approval=clf.predict(data_array)[0]
    result="Approved" if approval == 0 else "Rejected"

    #Risk Score Prediction

    risk_score=reg.predict(data_array)[0]
    
    # Generate SHAP explanations
    clf_shap_plot = generate_shap_plot(clf, data_array)
    reg_shap_plot = generate_shap_plot(reg, data_array)

    return render_template(
    "index.html",
    prediction_text=f"Loan Status: {result} | Risk Score: {risk_score:.2f}",
    clf_shap_plot=clf_shap_plot,
    reg_shap_plot=reg_shap_plot
)

if __name__== "__main__":
    app.run(debug=True)

