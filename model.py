import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
# Load the datasets
crop_data = pd.read_csv('Crop_recommendation (1).csv')
fertilizer_data = pd.read_csv('Fertilizer Prediction (1).csv')

# Drop 'Soil Type' column in fertilizer dataset
fertilizer_data = fertilizer_data.drop(columns=['Soil Type'])

# Encode categorical variables
label_encoder_crop = LabelEncoder()
crop_data['label'] = label_encoder_crop.fit_transform(crop_data['label'])

label_encoder_fertilizer = LabelEncoder()
fertilizer_data['Fertilizer Name'] = label_encoder_fertilizer.fit_transform(fertilizer_data['Fertilizer Name'])

# Define columns
numerical_cols_crop = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
numerical_cols_fertilizer = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temparature', 'Humidity ', 'Moisture']

# Split and scale data
X_crop = crop_data[numerical_cols_crop]
y_crop = crop_data['label']
X_crop_train, X_crop_test, y_crop_train, y_crop_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

X_fertilizer = fertilizer_data[numerical_cols_fertilizer]
y_fertilizer = fertilizer_data['Fertilizer Name']
X_fertilizer_train, X_fertilizer_test, y_fertilizer_train, y_fertilizer_test = train_test_split(X_fertilizer, y_fertilizer, test_size=0.2, random_state=42)

# Scale the data
scaler_crop = StandardScaler()
X_crop_train_scaled = scaler_crop.fit_transform(X_crop_train)

scaler_fertilizer = StandardScaler()
X_fertilizer_train_scaled = scaler_fertilizer.fit_transform(X_fertilizer_train)

# Train models
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_crop_train_scaled, y_crop_train)

fertilizer_model = RandomForestClassifier(n_estimators=100, random_state=42)
fertilizer_model.fit(X_fertilizer_train_scaled, y_fertilizer_train)

# Save models, scalers, and encoders using pickle
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(crop_model, f)
with open('fertilizer_model.pkl', 'wb') as f:
    pickle.dump(fertilizer_model, f)
with open('scaler_crop.pkl', 'wb') as f:
    pickle.dump(scaler_crop, f)
with open('scaler_fertilizer.pkl', 'wb') as f:
    pickle.dump(scaler_fertilizer, f)
with open('label_encoder_crop.pkl', 'wb') as f:
    pickle.dump(label_encoder_crop, f)
with open('label_encoder_fertilizer.pkl', 'wb') as f:
    pickle.dump(label_encoder_fertilizer, f)

print("Models, scalers, and encoders saved successfully!")