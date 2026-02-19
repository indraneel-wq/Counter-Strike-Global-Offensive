# 🎮 CS:GO Round Winner Predictor

This application predicts which team (Counter-Terrorists or Terrorists) is likely to win a round of CS:GO based on the current game state, such as health, money, armor, and equipment.

## 🚀 Features

*   **Real-time Prediction**: Leverages a machine learning model (Logistic Regression) trained on over 120,000 round snapshots.
*   **Key Factors**: Analyzes 20 critical features including weaponry count, players alive, grenade usage, and more.
*   **Simple Interface**: Fast and dark-themed UI for easy input.

## 📸 Screenshots

### Match Dashboard
![Dashboard](screenshots/dashboard.png)
*Interactive dashboard to input team health, economy, and loadouts.*

### Prediction Result
![Result Overlay](screenshots/result.png)
*Real-time prediction overlay showing the winning team and confidence level.*

## 🛠️ Tech Stack

*   **Backend**: Python, Flask
*   **Machine Learning**: Scikit-Learn, Pandas
*   **Frontend**: HTML5, CSS3, Bootstrap 5

## 📦 Data

The model uses the `csgo_round_snapshots.csv` dataset, which contains detailed information about thousands of competitive rounds.

## 🏁 How to Run

1.  **Clone this repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/csgo-winner-predictor.git
    cd csgo-winner-predictor
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**:
    ```bash
    python app.py
    ```

4.  **Open in Browser**:
    Navigate to `http://127.0.0.1:5002`

## ☁️ Deployment

This project includes a `Procfile` and is ready for deployment on platforms like Render or Heroku.

---
*Created by Indraneel*
