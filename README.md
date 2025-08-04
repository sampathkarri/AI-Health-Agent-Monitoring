🩺 AI Agent for Chronic Disease Monitoring
An intelligent AI agent designed to monitor risks for multiple chronic diseases, specifically Diabetes, Heart Disease, and Hypertension. This project utilizes a hybrid AI architecture, combining specialized predictive models with a powerful generative AI to provide users with a single, consolidated, and supportive health assessment.

The entire solution was built using IBM Cloud services and the IBM Granite foundation model.


✨ Key Features
Multi-Disease Monitoring: Simultaneously assesses risk for Diabetes, Heart Disease, and Hypertension.

Predictive Analysis: Uses three distinct machine learning models, built with AutoAI, for accurate, real-time risk prediction.

Generative AI Reasoning: Employs the IBM Granite LLM to interpret the predictions, prioritize risks, and generate personalized, easy-to-understand advice.

Interactive UI: A simple and clean web interface built with Gradio for user-friendly interaction.

⚙️ System Architecture
The application operates in a clear, end-to-end sequence:

Patient Data Input: A user enters their health metrics into the web interface.

Parallel Risk Assessment: The application sends the data to three separate, live API endpoints hosted on IBM Watson Machine Learning.

Predictive Analysis: Each of the three deployed models (Diabetes, Heart Disease, Hypertension) analyzes the data and returns a risk prediction (0 for Low Risk, 1 for High Risk).

Information Consolidation: The application gathers the three separate predictions.

Personalized Alert Generation: The predictions are formatted into a structured prompt and sent to the IBM Granite model, which generates a consolidated, human-readable response.

🛠️ Technology Stack
Platform: IBM watsonx.ai

Predictive Models: AutoAI (within Watson Studio)

Model Deployment: Watson Machine Learning

Generative Model: IBM Granite (via Prompt Lab)

Data Storage: Cloud Object Storage (COS)

Web Interface: Gradio

Deployment: IBM Cloud Code Engine

🚀 Local Setup and Installation
To run this project on your local machine, please follow these steps:

1. Clone the repository:

git clone https://github.com/sampathkarri/AI-Health-Agent-Monitoring
cd AI-Health-Agent

2. Create and activate a virtual environment:
python -m venv venv
.\venv\Scripts\activate

3. Install the required dependencies:

pip install -r requirements.txt

4. Set up your Environment Variables:
The application requires credentials to connect to IBM Cloud services. You must set these as environment variables. Copy the values from Cell 2 of your AI-Health-Agent-App (7).ipynb notebook.
# For Windows PowerShell
$env:IBM_CLOUD_API_KEY="your_api_key_here"
# ...and so on for all six variables

▶️ Usage
Once the setup is complete, run the application with:

python app.py

Open your web browser and navigate to the local URL provided by Gradio (usually http://127.0.0.1:7860).

☁️ Deployment
This application was deployed as a serverless web app on IBM Cloud Code Engine. The deployment was configured to build directly from the GitHub source code, with the credentials listed above configured as runtime environment variables within the Code Engine service for security.
