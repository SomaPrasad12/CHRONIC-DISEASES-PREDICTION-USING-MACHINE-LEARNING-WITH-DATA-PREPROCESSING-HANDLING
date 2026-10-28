from flask import Flask, render_template, request
import sqlite3
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

try:
    model = pickle.load(open('model.pkl', 'rb'))
except:
    model = None

def create_database():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT,
        prediction TEXT
    )
    ''')
    conn.commit()
    conn.close()

def train_model():
    X = np.array([
        [45, 130, 230],
        [50, 140, 250],
        [30, 120, 200],
        [60, 150, 300]
    ])
    y = np.array([1, 1, 0, 1])
    model = RandomForestClassifier()
    model.fit(X, y)
    pickle.dump(model, open('model.pkl', 'wb'))

@app.route('/')
def home():
    return '''
    <html>
    <body>
    <h2>Chronic Disease Prediction</h2>
    <form action="/predict" method="post">
    Age: <input type="text" name="age"><br><br>
    Blood Pressure: <input type="text" name="bp"><br><br>
    Cholesterol: <input type="text" name="chol"><br><br>
    <input type="submit" value="Predict">
    </form>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    if model:
        prediction = model.predict(final_features)
        output = prediction[0]
    else:
        output = "Model Not Loaded"
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO results (data, prediction) VALUES (?, ?)", 
                   (str(features), str(output)))
    conn.commit()
    conn.close()
    return f"<h2>Prediction: {output}</h2><a href='/'>Go Back</a>"

if __name__ == "__main__":
    create_database()
    train_model()
    app.run(debug=True)
