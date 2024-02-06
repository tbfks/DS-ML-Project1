import streamlit as st
import numpy as np

# Machine Learning
import joblib
import os

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
        
@st.cache_data
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

symptoms = [
    'Itching', 'Skin Rash', 'Nodal Skin Eruptions', 'Continuous Sneezing', 'Shivering',
    'Chills', 'Joint Pain', 'Stomach Pain', 'Acidity', 'Ulcers On Tongue',
    'Muscle Wasting', 'Vomiting', 'Burning Micturition', 'Spotting Urination',
    'Fatigue', 'Weight Gain', 'Anxiety', 'Cold Hands And Feet', 'Mood Swings',
    'Weight Loss', 'Restlessness', 'Lethargy', 'Patches In Throat',
    'Irregular Sugar Level', 'Cough', 'High Fever', 'Sunken Eyes', 'Breathlessness',
    'Sweating', 'Dehydration', 'Indigestion', 'Headache', 'Yellowish Skin',
    'Dark Urine', 'Nausea', 'Loss Of Appetite', 'Pain Behind The Eyes',
    'Back Pain', 'Constipation', 'Abdominal Pain', 'Diarrhoea', 'Mild Fever',
    'Yellow Urine', 'Yellowing Of Eyes', 'Acute Liver Failure', 'Fluid Overload',
    'Swelling Of Stomach', 'Swelled Lymph Nodes', 'Malaise', 'Blurred And Distorted Vision',
    'Phlegm', 'Throat Irritation', 'Redness Of Eyes', 'Sinus Pressure', 'Runny Nose',
    'Congestion', 'Chest Pain', 'Weakness In Limbs', 'Fast Heart Rate',
    'Pain During Bowel Movements', 'Pain In Anal Region', 'Bloody Stool',
    'Irritation In Anus', 'Neck Pain', 'Dizziness', 'Cramps', 'Bruising', 'Obesity',
    'Swollen Legs', 'Swollen Blood Vessels', 'Puffy Face And Eyes', 'Enlarged Thyroid',
    'Brittle Nails', 'Swollen Extremities', 'Excessive Hunger', 'Extra Marital Contacts',
    'Drying And Tingling Lips', 'Slurred Speech', 'Knee Pain', 'Hip Joint Pain',
    'Muscle Weakness', 'Stiff Neck', 'Swelling Joints', 'Movement Stiffness',
    'Spinning Movements', 'Loss Of Balance', 'Unsteadiness', 'Weakness Of One Body Side',
    'Loss Of Smell', 'Bladder Discomfort', 'Foul Smell Of Urine',
    'Continuous Feel Of Urine', 'Passage Of Gases', 'Internal Itching',
    'Toxic Look (Typhos)', 'Depression', 'Irritability', 'Muscle Pain',
    'Altered Sensorium', 'Red Spots Over Body', 'Belly Pain', 'Abnormal Menstruation',
    'Dischromic Patches', 'Watering From Eyes', 'Increased Appetite', 'Polyuria',
    'Family History', 'Mucoid Sputum', 'Rusty Sputum', 'Lack Of Concentration',
    'Visual Disturbances', 'Receiving Blood Transfusion', 'Receiving Unsterile Injections',
    'Coma', 'Stomach Bleeding', 'Distention Of Abdomen', 'History Of Alcohol Consumption',
    'Fluid Overload', 'Blood In Sputum', 'Prominent Veins On Calf', 'Palpitations',
    'Painful Walking', 'Pus Filled Pimples', 'Blackheads', 'Scurring', 'Skin Peeling',
    'Silver Like Dusting', 'Small Dents In Nails', 'Inflammatory Nails', 'Blister',
    'Red Sore Around Nose', 'Yellow Crust Ooze'
]


# symptoms_to_show = [symptom.title().replace("_", " ") for symptom in symptoms]

def run_ml_app():
    st.subheader("Select the symptoms below:")

    selected_symptoms = st.multiselect(
        "Please select the symptom:", 
        options = symptoms,
        placeholder = "Select the symptoms..."
    )

    def encoded_symptoms(selected_symptoms, all_symptoms):
        """
        Encodes user-selected symptoms into a one-hot encoded vector for logistic regression.

        Args:
            selected_symptoms (list): List of symptom names selected by the user.
            all_symptoms (list): List of all symptom names used in the model.

        Returns:
            numpy.ndarray: Encoded symptom vector as a 1D NumPy array.
        """

        encoded_vector = np.zeros(len(all_symptoms))
        for symptom in selected_symptoms:
            if symptom in all_symptoms:
                encoded_vector[all_symptoms.index(symptom)] = 1

        # Reshape into a 2D array with one sample (single prognosis)
        encoded_vector = encoded_vector.reshape(1, -1)

        return encoded_vector
    
    model = load_model("model_lr.pkl")
    user_symptoms = selected_symptoms
    all_symptoms = symptoms  # Use your actual symptom list here

    encoded_vector = encoded_symptoms(user_symptoms, all_symptoms)

    prediction = model.predict(encoded_vector)
    confi = model.predict_proba(encoded_vector)
    max_confi = max(confi[0])

    # Make a Generate button
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    generate_button = st.button("Generate Prediction")

    # Only execute prediction logic if the button is clicked
    if generate_button:
        st.session_state.clicked = True   

        # Show the prediction
        st.write(f"Predicted disease: {prediction}")
        st.write(f"Max Confidence: {(max_confi) * 100} %")