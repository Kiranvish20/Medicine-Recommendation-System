from flask import Flask, request, redirect, url_for, render_template, session, jsonify
from difflib import get_close_matches
import mysql.connector
import numpy as np
import pandas as pd
import pickle
import os
import smtplib
from email.message import EmailMessage
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail 
from flask_bcrypt import Bcrypt # for password hashing

app = Flask(__name__)
app.secret_key = "" 

bcrypt = Bcrypt(app) 
s = URLSafeTimedSerializer(app.secret_key, salt='password-reset')

# ---------------------- Database Connection ----------------------
db = mysql.connector.connect(
    host="localhost",
    port=3307,     # Update this if your MySQL server uses a different port
    user="root",
    password="1234",
    database="medicine_DB"
)
cursor = db.cursor()

# --- Fuzzy match function for backend ---
def correct_symptom(symptom):
    """Return closest valid symptom from symptoms_dict if available"""
    symptom = symptom.lower().replace(" ", "_")
    matches = get_close_matches(symptom, symptoms_dict.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

# ---------------------- Load Dataset & Model ----------------------
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

svc = pickle.load(open('models/svc.pkl','rb'))

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# ---------------------- Helper Functions ----------------------
def helper(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'])
    pre = [col for col in precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values]
    med = [med for med in medications[medications['Disease'] == dis]['Medication'].values]
    die = [die for die in diets[diets['Disease'] == dis]['Diet'].values]
    wrkout = workout[workout['disease'] == dis]['workout']
    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# ---------------------- Authentication Routes ----------------------

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard')) # ✅ redirect to dashboard if logged in
    return render_template('index.html') 

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # HASH THE PASSWORD BEFORE STORING IT
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        try:
            # Use the hashed_password in the query
            cursor.execute("INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
                           (username, email, hashed_password))
            db.commit()
            return redirect(url_for('signin'))
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('signup.html')


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user:
            # CHECK THE HASHED PASSWORD
            stored_hashed_password = user[3] # Assuming password_hash is the 4th column
            if bcrypt.check_password_hash(stored_hashed_password, password):
                session['user_id'] = user[0]
                session['username'] = user[1]
                return redirect(url_for('dashboard'))
            else:
                return "Invalid credentials. <a href='/'>Try again</a>"
        else:
            return "Invalid credentials. <a href='/'>Try again</a>"
    return render_template('signin.html')

# ---------------------- NEW CODE FOR FORGOT PASSWORD ----------------------

# --- CONFIGURATION (UPDATE THESE) ---
os.environ['SENDGRID_API_KEY'] = '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'
sender_email = 'kiranv0303@gmail.com' 
your_website_domain = 'http://127.0.0.1:5000'

@app.route('/forgot', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')

        # Check if email exists in the database
        cursor.execute("SELECT id, name FROM users WHERE email = %s", (email,))
        user_info = cursor.fetchone()

        if user_info:
            user_id, username = user_info
            token = s.dumps(str(user_id), salt='password-reset')

            try:
                # Store the token in the new password_resets table
                cursor.execute(
                    "INSERT INTO password_resets (user_id, token) VALUES (%s, %s)",
                    (user_id, token)
                )
                db.commit()

                reset_url = f"{your_website_domain}/reset_password?token={token}"

                # Send the password reset email using SendGrid
                message = Mail(
                    from_email=sender_email,
                    to_emails=email,
                    subject='Password Reset Request',
                    html_content=f"Hello {username},<br><br>"
                                 f"You have requested a password reset. Please click the link below to reset your password:<br><br>"
                                 f"<a href='{reset_url}'>{reset_url}</a><br><br>"
                                 f"This link will expire in 1 hour.<br><br>"
                                 f"If you did not request this, please ignore this email."
                )
                sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
                response = sg.send(message)

                print(f"Password reset email sent to {email}. Status code: {response.status_code}")

            except Exception as e:
                print(f"Failed to send email or save token: {e}")
                db.rollback()
                return "Failed to send reset link. Please try again later.", 500

        # Always show a generic success message
        return "If your email is registered, a password reset link has been sent to your inbox.", 200
    
    return render_template('forgot.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    # Handle the GET request to show the reset form
    if request.method == 'GET':
        token = request.args.get('token')
        
        if not token:
            return "Missing token.", 400
        
        try:
            # Check if the token is valid (e.g., within 1 hour)
            user_id_str = s.loads(token, salt='password-reset', max_age=3600)
            
            # Check if the token exists in our new database table
            cursor.execute("SELECT user_id FROM password_resets WHERE token = %s", (token,))
            db_user_id = cursor.fetchone()

            if not db_user_id or str(db_user_id[0]) != user_id_str:
                return "The password reset link is invalid.", 400
            
            # Store the user ID and token in the session for the POST request
            session['reset_user_id'] = user_id_str
            session['reset_token'] = token

            # Render the reset password HTML page
            return render_template('reset_password.html')
        
        except SignatureExpired:
            return "The password reset link has expired. Please request a new one.", 400
        except Exception as e:
            print(e)
            return "The password reset link is invalid.", 400

    # Handle the POST request to update the password
    elif request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        user_id_str = session.get('reset_user_id')
        token = session.get('reset_token')
        
        if not user_id_str or not token:
            return "Session expired. Please request a new reset link.", 400
        
        if new_password != confirm_password:
            return "Passwords do not match."

        # Delete the token from the database to prevent reuse
        try:
            cursor.execute("DELETE FROM password_resets WHERE token = %s", (token,))
            db.commit()
        except Exception as e:
            print(f"Error deleting token: {e}")
            db.rollback()

        # Update the password in the users table
        try:
            # HASH THE NEW PASSWORD BEFORE STORING IT
            hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
            
            cursor.execute("UPDATE users SET password_hash = %s WHERE id = %s", 
                           (hashed_password, user_id_str))
            db.commit()

            # Clear the session variables
            session.pop('reset_user_id', None)
            session.pop('reset_token', None)

            return "Your password has been reset successfully. You can now <a href='/signin'>sign in</a>."
        except Exception as e:
            print(f"Error updating password: {e}")
            db.rollback()
            return "An error occurred while updating the password.", 500

# ---------------------- Logout Route ----------------------
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('home'))

# ---------------------- Dashboard & Prediction ----------------------

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('home'))

    user_id = session['user_id']
    username = session['username']
    
    # Get the first letter of the username
    user_initial = username[0].upper()
    
    # Fetch prediction history from the database
    cursor.execute("SELECT symptoms, disease, prediction_date FROM prediction_history WHERE user_id = %s ORDER BY prediction_date DESC", (user_id,))
    history = cursor.fetchall()
    
    # Convert history to a list of dictionaries for easier use in Jinja2
    prediction_history = []
    for item in history:
        prediction_history.append({
            'symptoms': item[0],
            'disease': item[1],
            'date': item[2].strftime('%Y-%m-%d %H:%M') # Format the date for display
        })

    return render_template('dashboard.html', 
                           prediction_history=prediction_history,
                           user_initial=user_initial) # Pass the initial to the template

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('home'))

    symptoms = request.form.get('symptoms')
    user_id = session['user_id']
    
    # Add these lines to get user info
    username = session.get('username')
    user_initial = username[0].upper()
    
    if not symptoms or symptoms.strip() == "":
        return redirect(url_for('dashboard'))

    raw_symptoms = [s.strip() for s in symptoms.split(',')]
    user_symptoms = []
    for s in raw_symptoms:
        corrected = correct_symptom(s)
        if corrected:
            user_symptoms.append(corrected)

    if not user_symptoms:
        return redirect(url_for('dashboard'))

    predicted_disease = get_predicted_value(user_symptoms)
    dis_des, precautions_list, meds, rec_diet, wrkout = helper(predicted_disease)
    my_precautions = [i for i in precautions_list[0]]

    try:
        symptoms_str = ", ".join(user_symptoms)
        cursor.execute(
            "INSERT INTO prediction_history (user_id, symptoms, disease) VALUES (%s, %s, %s)",
            (user_id, symptoms_str, predicted_disease)
        )
        db.commit()
    except Exception as e:
        print(f"Error saving prediction history: {e}") 
    
    # Pass user_initial to the template here
    return render_template('dashboard.html',
                           predicted_disease=predicted_disease,
                           dis_des=dis_des,
                           my_precautions=my_precautions,
                           medications=meds,
                           my_diet=rec_diet,
                           workout=wrkout,
                           user_initial=user_initial) # Pass user_initial

# ---------------------- Prediction History ----------------------

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('home'))

    user_id = session['user_id']
    username = session['username']
    user_initial = username[0].upper()

    # Fetch prediction history from the database, including the 'id'
    cursor.execute("SELECT id, symptoms, disease, prediction_date FROM prediction_history WHERE user_id = %s ORDER BY prediction_date DESC", (user_id,))
    history_data = cursor.fetchall()

    prediction_history = []
    for item in history_data:
        prediction_history.append({
            'id': item[0], # Include the ID for deletion
            'symptoms': item[1],
            'disease': item[2],
            'date': item[3].strftime('%Y-%m-%d %H:%M')
        })

    return render_template('profile.html', 
                           user_initial=user_initial,
                           prediction_history=prediction_history)

# ---------------------- Delete Prediction Route ----------------------

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('home'))

    user_id = session['user_id']
    
    try:
        # Ensure the user owns the prediction before deleting
        cursor.execute("DELETE FROM prediction_history WHERE id = %s AND user_id = %s", (prediction_id, user_id))
        db.commit()
    except Exception as e:
        print(f"Error deleting prediction: {e}")

    return redirect(url_for('profile')) # Redirect back to the profile page




# ---------------------- Other Pages ----------------------

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

# ---------------------- Run App ----------------------

if __name__ == '__main__':

    app.run(debug=True)
