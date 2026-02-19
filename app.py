from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib
import os

app = Flask(__name__)

# Global variables
model = None
scaler = None
label_encoder = None
feature_columns = [
    'time_left', 'ct_score', 't_score', 'map', 'bomb_planted', 'ct_health',
    't_health', 'ct_armor', 't_armor', 'ct_money', 't_money', 'ct_helmets',
    't_helmets', 'ct_defuse_kits', 'ct_players_alive', 't_players_alive',
    'ct_weapon_ak47', 't_weapon_ak47', 'ct_weapon_aug', 't_weapon_aug',
    'ct_weapon_awp', 't_weapon_awp', 'ct_weapon_bizon', 't_weapon_bizon',
    'ct_weapon_cz75auto', 't_weapon_cz75auto', 'ct_weapon_elite',
    't_weapon_elite', 'ct_weapon_famas', 't_weapon_famas',
    'ct_weapon_g3sg1', 't_weapon_g3sg1', 'ct_weapon_galilar',
    't_weapon_galilar', 'ct_weapon_glock', 't_weapon_glock',
    'ct_weapon_m249', 't_weapon_m249', 'ct_weapon_m4a1s', 't_weapon_m4a1s',
    'ct_weapon_m4a4', 't_weapon_m4a4', 'ct_weapon_mac10', 't_weapon_mac10',
    'ct_weapon_mag7', 't_weapon_mag7', 'ct_weapon_mp5sd', 't_weapon_mp5sd',
    'ct_weapon_mp7', 't_weapon_mp7', 'ct_weapon_mp9', 't_weapon_mp9',
    'ct_weapon_negev', 't_weapon_negev', 'ct_weapon_nova', 't_weapon_nova',
    'ct_weapon_p2000', 't_weapon_p2000', 'ct_weapon_p250', 't_weapon_p250',
    'ct_weapon_p90', 't_weapon_p90', 'ct_weapon_sawedoff',
    't_weapon_sawedoff', 'ct_weapon_scar20', 't_weapon_scar20',
    'ct_weapon_sg553', 't_weapon_sg553', 'ct_weapon_ssg08',
    't_weapon_ssg08', 'ct_weapon_ump45', 't_weapon_ump45',
    'ct_weapon_usps', 't_weapon_usps', 'ct_weapon_xm1014',
    't_weapon_xm1014', 'ct_weapon_deagle', 't_weapon_deagle',
    'ct_weapon_fiveseven', 't_weapon_fiveseven', 'ct_weapon_usps',
    't_weapon_usps', 'ct_weapon_p250', 't_weapon_p250', 'ct_weapon_p2000',
    't_weapon_p2000', 'ct_weapon_tec9', 't_weapon_tec9',
    'ct_grenade_hegrenade', 't_grenade_hegrenade', 'ct_grenade_flashbang',
    't_grenade_flashbang', 'ct_grenade_smokegrenade',
    't_grenade_smokegrenade', 'ct_grenade_incendiarygrenade',
    't_grenade_incendiarygrenade', 'ct_grenade_molotovgrenade',
    't_grenade_molotovgrenade', 'ct_grenade_decoygrenade',
    't_grenade_decoygrenade'
]

# Only use the top 20 features for input simplicity, but keep whole dataset structure if needed.
# For simplicity, we will retrain on just the top 20 features identified in the notebook.
top_20_features = [
    't_armor', 't_weapon_ak47', 'ct_armor', 't_weapon_sg553', 'ct_weapon_m4a4', 
    'ct_health', 't_players_alive', 't_health', 'ct_weapon_awp', 'bomb_planted', 
    't_weapon_awp', 't_grenade_smokegrenade', 'ct_players_alive', 'ct_money', 
    'ct_weapon_aug', 'ct_weapon_sg553', 't_grenade_flashbang', 'ct_weapon_ak47', 
    't_weapon_glock', 't_money'
]

def train_model():
    global model, scaler, label_encoder
    
    try:
        if os.path.exists('model.pkl'):
            print("Loading existing model...")
            model = joblib.load('model.pkl')
            scaler = joblib.load('scaler.pkl')
            label_encoder = joblib.load('encoder.pkl')
            return True

        if not os.path.exists('csgo_round_snapshots.csv'):
            print("Dataset not found!")
            return False

        print("Loading dataset...")
        df = pd.read_csv('csgo_round_snapshots.csv')
        
        # Preprocessing
        le = LabelEncoder()
        df['bomb_planted'] = le.fit_transform(df['bomb_planted'])
        label_encoder = le # Save encoder to decode/encode inputs if needed
        
        # Map target
        # Assuming last column is 'round_winner' based on common dataset structure
        # Let's check from the notebook outline, it seemed to imply checking columns
        # The notebook had 'round_winner' as target usually.
        # Let's assume it is 'round_winner' and decode it: CT=0, T=1 or similar.
        # Actually in the notebook, it seems 'round_winner' is the target.
        if 'round_winner' in df.columns:
             # Encode target: CT -> 0, T -> 1 (or vice versa, let's stick to standard)
             # Notebook used LabelEncoder on it probably.
             le_target = LabelEncoder()
             df['round_winner'] = le_target.fit_transform(df['round_winner'])
             y = df['round_winner']
        else:
             print("Target column 'round_winner' not found.")
             return False

        # Use only top 20 features
        X = df[top_20_features]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train (Using Logistic Regression as baseline, fast and good enough)
        print("Training model...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Save
        joblib.dump(model, 'model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoder, 'encoder.pkl') # For bomb_planted
        
        print("Model trained.")
        return True

    except Exception as e:
        print(f"Error training model: {e}")
        return False

@app.route('/')
def index():
    if not model:
        train_model()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not trained'}), 500
    
    try:
        data = request.json
        input_data = []
        
        # Order matters! Must match top_20_features list
        # We need to handle 'bomb_planted' carefully (True/False or 1/0)
        
        for feature in top_20_features:
            val = data.get(feature)
            if feature == 'bomb_planted':
                # Expect boolean or string "True"/"False" from frontend
                val = 1 if str(val).lower() == 'true' else 0
            else:
                 val = float(val)
            input_data.append(val)
            
        # Scale
        input_scaled = scaler.transform([input_data])
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]
        
        # Decode prediction (0->CT, 1->T usually with LabelEncoder sorted alph.)
        # CT comes before T alphbetically.
        winner = "CT" if prediction == 0 else "T"
        confidence = prob[prediction] * 100
        
        return jsonify({
            'winner': winner,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    train_model()
    app.run(debug=True, port=5002)
