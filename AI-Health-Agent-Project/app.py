# app.py
import os
from flask import Flask, request, jsonify, render_template

from ibm_watson_machine_learning import APIClient
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.client import APIClient as WatsonxAPIClient

# ==============================================================================
#                      --- PASTE YOUR CREDENTIALS HERE ---
# ==============================================================================
API_KEY = "PASTE_YOUR_FINAL_PROJECT_KEY_HERE"
PROJECT_ID = "PASTE_YOUR_WATSONX_PROJECT_ID_HERE"
SPACE_ID = "PASTE_YOUR_DEPLOYMENT_SPACE_ID_HERE"
DIABETES_DEPLOYMENT_ID = "PASTE_YOUR_DIABETES_DEPLOYMENT_ID_HERE"
HEART_DEPLOYMENT_ID = "PASTE_YOUR_HEART_DISEASE_DEPLOYMENT_ID_HERE"
HYPERTENSION_DEPLOYMENT_ID = "PASTE_YOUR_HYPERTENSION_DEPLOYMENT_ID_HERE"
# ==============================================================================

# --- Connect to IBM Cloud Services ---
REGION_URL = "https://eu-gb.ml.cloud.ibm.com" 

wml_credentials = {"url": REGION_URL, "apikey": API_KEY}
wml_client = APIClient(wml_credentials)
wml_client.set.default_space(SPACE_ID)

watsonx_credentials = {"url": REGION_URL, "apikey": API_KEY}
watsonx_client = WatsonxAPIClient(credentials=watsonx_credentials, project_id=PROJECT_ID)

app = Flask(__name__)

master_prompt = """
You are an AI Health Assistant. Your job is to provide a single, consolidated, and supportive message to a user based on their health risk factors.
**Instructions:**
1. Review all three risk levels: Diabetes, Heart Condition, and Hypertension.
2. Identify any risk that is "High".
3. If multiple risks are "High", prioritize the Heart Condition risk as the main topic.
4. If all risks are "Low", provide a general wellness tip.
5. For each "High" risk, provide one simple, actionable lifestyle tip.
6. The tone must be calm, supportive, and never alarming.
7. **Crucially, always end the response by strongly recommending that the user consult a real doctor for personalized medical advice.**
**User's Health Data Analysis:**
- Diabetes Risk: {diabetes_risk}
- Heart Condition Risk: {heart_risk}
- Hypertension Risk: {hypertension_risk}
**AI Assistant's Response:**
"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # --- Prepare inputs from the simple form ---
        # NOTE: We use simplified inputs here. A full production app would have a much larger form.
        diabetes_input = { "input_data": [{ "fields": ["Glucose", "BMI", "Age"], "values": [[float(data['glucose']), float(data['bmi']), int(data['age'])]] }] }
        heart_disease_input = { "input_data": [{ "fields": ["id", "age", "sex", "dataset", "cp"], "values": [[1, int(data['age']), 1, "Cleveland", "asymptomatic"]] }] }
        hypertension_input = { "input_data": [{ "fields": ["age", "sex", "cp", "trtbps"], "values": [[int(data['age']), 1, 3, 145]] }] }

        # --- Call Predictive Models ---
        diabetes_pred = wml_client.deployments.score(DIABETES_DEPLOYMENT_ID, diabetes_input)['predictions'][0]['values'][0][0]
        heart_pred = wml_client.deployments.score(HEART_DEPLOYMENT_ID, heart_disease_input)['predictions'][0]['values'][0][0]
        hypertension_pred = wml_client.deployments.score(HYPERTENSION_DEPLOYMENT_ID, hypertension_input)['predictions'][0]['values'][0][0]

        diabetes_risk_text = "High" if diabetes_pred == 1 else "Low"
        heart_risk_text = "High" if heart_pred == 1 else "Low"
        hypertension_risk_text = "High" if hypertension_pred == 1 else "Low"

        # --- Call the Granite AI Agent ---
        final_prompt = master_prompt.format(diabetes_risk=diabetes_risk_text, heart_risk=heart_risk_text, hypertension_risk=hypertension_risk_text)
        model_params = {"decoding_method": "greedy", "max_new_tokens": 300}
        granite_model = Model(model_id="ibm/granite-13b-instruct-v2", params=model_params, client=watsonx_client)
        generated_response = granite_model.generate_text(prompt=final_prompt)

        return jsonify({'prediction_text': generated_response})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)