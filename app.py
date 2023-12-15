from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'votre_clé_secrète'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    salaire = db.Column(db.Float)
    prediction_result = db.Column(db.String)

# Charger le modèle et le StandardScaler
best_model = joblib.load('best_naive_bayes_model.joblib')
sc = joblib.load('standard_scaler.joblib')

# Liste d'utilisateurs autorisés
utilisateurs_autorises = [{'username': 'admin@gmail.com', 'password': 'admin2023'}]

@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    result = None

    if request.method == 'POST':
        age = int(request.form['age'])
        salaire = float(request.form['salaire'])

        # Prétraiter les données comme lors de l'entraînement du modèle
        data = np.array([[age, salaire]])
        data = sc.transform(data)  # Utiliser le même StandardScaler que celui utilisé lors de l'entraînement

        # Utiliser le modèle pour faire une prédiction
        prediction = best_model.predict(data)

        # Enregistrer les informations dans la base de données
        nouvelle_prediction = Prediction(age=age, salaire=salaire, prediction_result=str(prediction[0]))
        db.session.add(nouvelle_prediction)
        db.session.commit()

        # Assurez-vous d'adapter cela à la sortie spécifique de votre modèle (1 ou 0, par exemple)
        result = "Produit acheté" if prediction[0] == 1 else "Produit non acheté"

    return render_template('index.html', result=result)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Vérifiez les informations d'identification
        username = request.form['username']
        password = request.form['password']

        # Exemple simple de vérification (à adapter selon vos besoins)
        for utilisateur in utilisateurs_autorises:
            if username == utilisateur['username'] and password == utilisateur['password']:
                session['logged_in'] = True  # Stocke l'état de connexion dans la session
                return redirect(url_for('index'))  # Redirige vers la page principale si les informations d'identification sont correctes

    return render_template('login.html')  # Affiche la page de connexion

@app.route('/visualiser-donnees', methods=['GET'])
def visualiser_donnees():
    donnees = Prediction.query.all()
    return render_template('visualiser_donnees.html', donnees=donnees)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
