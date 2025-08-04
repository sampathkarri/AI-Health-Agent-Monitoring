import os
import gradio as gr
from ibm_watson_machine_learning import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference

# --- Step 1: Hardcoded Project Credentials ---
# All values are taken directly from the project notebook.
API_KEY = "CogOqUMsBHlotDis07P5UG0vkANcsB7JNCtCzTRPMkZ6"
PROJECT_ID = "71124e53-992d-46ce-8e3b-1ade2c10caba"
SPACE_ID = "f210a915-20f4-462f-a4b3-3ca197b32a41"
DIABETES_DEPLOYMENT_ID = "4a16a588-86ae-40e9-92a8-bfe85957d205"
HEART_DEPLOYMENT_ID = "1350a17d-e7c5-4c1f-9ca7-d5f173c4dcfd"
HYPERTENSION_DEPLOYMENT_ID = "8b6ab145-9d92-47ae-8a73-408a7b09a57a"

# --- Step 2: Setup IBM Cloud API Clients ---
LONDON_URL = "https://eu-gb.ml.cloud.ibm.com"
wml_credentials = {"url": LONDON_URL, "apikey": API_KEY}
wml_client = APIClient(wml_credentials)
wml_client.set.default_space(SPACE_ID)
watsonx_credentials = {"url": LONDON_URL, "apikey": API_KEY}

# --- Step 3: Define the Core Function for the AI Agent ---
def get_health_assessment(
    # Diabetes inputs
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age,
    # Heart/Hypertension inputs
    sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):

    try:
        # --- Format data for each predictive model ---
        diabetes_fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        diabetes_values = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
        diabetes_input = {"input_data": [{"fields": diabetes_fields, "values": diabetes_values}]}

        cardiac_fields = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        cardiac_values = [[age, (1 if sex=="Male" else 0), cp, trestbps, chol, (1 if fbs=="True" else 0), restecg, thalach, (1 if exang=="True" else 0), oldpeak, slope, ca, thal]]
        cardiac_input = {"input_data": [{"fields": cardiac_fields, "values": cardiac_values}]}

        # --- Call the deployed models ---
        print("Scoring predictive models...")
        diabetes_pred = wml_client.deployments.score(DIABETES_DEPLOYMENT_ID, diabetes_input)
        heart_pred = wml_client.deployments.score(HEART_DEPLOYMENT_ID, cardiac_input)
        hypertension_pred = wml_client.deployments.score(HYPERTENSION_DEPLOYMENT_ID, cardiac_input)
        
        diabetes_result = diabetes_pred['predictions'][0]['values'][0][0]
        heart_result = heart_pred['predictions'][0]['values'][0][0]
        hypertension_result = hypertension_pred['predictions'][0]['values'][0][0]

        # --- Build the prompt for the Granite model ---
        diabetes_risk = "High" if diabetes_result == 1 else "Low"
        heart_risk = "High" if heart_result == 1 else "Low"
        hypertension_risk = "High" if hypertension_result == 1 else "Low"
        
        prompt = f"""You are an AI Health Assistant. Your job is to provide a single, consolidated, and supportive message to a user based on their health risk factors.
        **User's Health Data Analysis:**
        - Diabetes Risk: {diabetes_risk}
        - Heart Condition Risk: {heart_risk}
        - Hypertension Risk: {hypertension_risk}
        **Instructions:**
        1. Review all three risk levels. 2. Identify any "High" risk. 3. If multiple risks are "High", choose Heart Condition as the primary topic. 4. For each "High" risk, provide one simple, actionable lifestyle tip. 5. The tone must be calm and supportive. 6. **Crucially, always end by strongly recommending the user consult a real doctor.**
        **AI Assistant's Response:**
        """

        # --- Call the Granite model to generate the final advice ---
        print("Generating final advice from Granite model...")
        model_params = {"decoding_method": "greedy", "max_new_tokens": 300, "min_new_tokens": 50}
        granite_model = ModelInference(model_id="ibm/granite-3-3-8b-instruct", params=model_params, credentials=watsonx_credentials, project_id=PROJECT_ID)
        generated_response = granite_model.generate_text(prompt=prompt)
        
        return generated_response

    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- Step 4: Build the Gradio Web Interface with Pre-filled Values ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Health Agent") as iface:
    gr.Markdown("# 🩺 AI Agent for Chronic Disease Monitoring")
    gr.Markdown("Enter patient health metrics to get a personalized risk assessment from the AI agent. The form is pre-filled with sample data.")
    
    with gr.Accordion("Patient Details", open=True):
        with gr.Row():
            age = gr.Number(label="Age", value=56)
            sex = gr.Radio(label="Sex", choices=["Male", "Female"], value="Male")
            pregnancies = gr.Number(label="Pregnancies", value=6)
    
    with gr.Accordion("Health Metrics", open=True):
        with gr.Row():
            glucose = gr.Number(label="Glucose", value=148)
            blood_pressure = gr.Number(label="Diastolic Blood Pressure", value=72)
            trestbps = gr.Number(label="Resting Systolic BP (trestbps)", value=120)
        with gr.Row():
            chol = gr.Number(label="Serum Cholesterol (chol)", value=236)
            skin_thickness = gr.Number(label="Skin Thickness", value=35)
            insulin = gr.Number(label="Insulin", value=0)
            bmi = gr.Number(label="BMI", value=33.6)
            dpf = gr.Number(label="Diabetes Pedigree Func", value=0.627)
    
    with gr.Accordion("Advanced Cardiac Metrics", open=False):
        cp = gr.Slider(label="Chest Pain Type (cp)", minimum=0, maximum=3, step=1, value=0)
        fbs = gr.Radio(label="Fasting Blood Sugar > 120 mg/dl (fbs)", choices=["True", "False"], value="False")
        restecg = gr.Slider(label="Resting ECG (restecg)", minimum=0, maximum=2, step=1, value=1)
        thalach = gr.Number(label="Max Heart Rate Achieved (thalach)", value=178)
        exang = gr.Radio(label="Exercise Induced Angina (exang)", choices=["True", "False"], value="False")
        oldpeak = gr.Number(label="Oldpeak", value=0.8)
        slope = gr.Slider(label="Slope", minimum=0, maximum=2, step=1, value=2)
        ca = gr.Slider(label="Major Vessels Colored (ca)", minimum=0, maximum=4, step=1, value=0)
        thal = gr.Slider(label="Thal", minimum=0, maximum=3, step=1, value=2)

    btn = gr.Button("Get Health Assessment", variant="primary")
    output_text = gr.Textbox(label="AI Agent's Response", lines=6)

    btn.click(fn=get_health_assessment, 
              inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal], 
              outputs=output_text)

if __name__ == "__main__":
    iface.launch()