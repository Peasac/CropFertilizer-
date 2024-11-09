import streamlit as st
import pandas as pd
import pickle
import base64

# Load models and encoders (assuming you already have your .pkl files saved)
with open('crop_model.pkl', 'rb') as f:
    crop_model = pickle.load(f)
with open('fertilizer_model.pkl', 'rb') as f:
    fertilizer_model = pickle.load(f)
with open('scaler_crop.pkl', 'rb') as f:
    scaler_crop = pickle.load(f)
with open('scaler_fertilizer.pkl', 'rb') as f:
    scaler_fertilizer = pickle.load(f)
with open('label_encoder_crop.pkl', 'rb') as f:
    label_encoder_crop = pickle.load(f)
with open('label_encoder_fertilizer.pkl', 'rb') as f:
    label_encoder_fertilizer = pickle.load(f)

# Function to encode image to Base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Get base64 of the background image
img_file = 'nightcrop.jpg'  # Ensure this path is correct
base64_img = get_base64_of_bin_file(img_file)

# Custom CSS for background image, footer, and button styles
st.markdown(
    f"""
    <style>
    /* Apply background image */
    .stApp {{
        background: url("data:image/jpg;base64,{base64_img}") no-repeat center center fixed;
        background-size: cover;
        color: white;
        padding-bottom: 120px;  /* Adds space above footer */
    }}
    /* Ensure all text is white in both light and dark modes */
    .stText, .stTitle, .stHeader, .stSubheader {{
        color: white !important;
    }}
    /* Ensure all text labels for inputs are clearly visible */
    label {{
        color: #000000 !important;  /* Dark color for readability */
        font-weight: bold;
        background-color: rgba(255, 255, 255, 0.7);  /* Semi-transparent background */
        padding: 5px;
        border-radius: 5px;
    }}
    /* Make the input fields clearly visible */
    .stNumberInput, .stTextInput {{
        background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent background for inputs */
        color: black !important;  /* Black text inside the input fields */
        border-radius: 5px;
    }}
    /* Ensure the buttons are visible */
    .stButton button {{
        color: black !important;  /* Black text for buttons */
        background-color: #F0F0F0 !important;  /* Light background for buttons */
        border-radius: 5px;
        font-weight: bold;
    }}
    .footer {{
        width: 100%;
        background-color: #1F2937;
        color: white;
        text-align: center;
        padding: 10px;
        position: relative;
        margin-top: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("Crop and Fertilizer Recommendation System")

st.header("Enter values for recommendation:")

# Input fields for common parameters
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorous (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)

# Crop-specific inputs
ph = st.number_input("pH level", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# Fertilizer-specific input
moisture = st.number_input("Moisture (%)", min_value=0.0)

# Recommend Crop
if st.button("Recommend Crop"):
    input_data_crop = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    input_data_crop_scaled = scaler_crop.transform(input_data_crop)
    crop_label = crop_model.predict(input_data_crop_scaled)
    crop_name = label_encoder_crop.inverse_transform(crop_label)[0]
    st.success(f"Recommended Crop: {crop_name}")

# Recommend Fertilizer
if st.button("Recommend Fertilizer"):
    input_data_fertilizer = pd.DataFrame([[N, P, K, temperature, humidity, moisture]], columns=['Nitrogen', 'Phosphorous', 'Potassium', 'Temparature', 'Humidity ', 'Moisture'])
    input_data_fertilizer_scaled = scaler_fertilizer.transform(input_data_fertilizer)
    fertilizer_label = fertilizer_model.predict(input_data_fertilizer_scaled)
    fertilizer_name = label_encoder_fertilizer.inverse_transform(fertilizer_label)[0]
    st.success(f"Recommended Fertilizer: {fertilizer_name}")

# Footer section
st.markdown(
    """
    <div class="footer">
        Copyright © SmartSow2024
    </div>
    """,
    unsafe_allow_html=True
)
