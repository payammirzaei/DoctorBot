from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Expanded Training Data: Symptom-based queries and their responses
training_data = [
    "I have a fever", "fever",
    "I feel cold", "cold",
    "I have a headache", "headache",
    "I have a cough", "cough",
    "I feel very tired and fatigued", "fatigue",
    "I have a sore throat", "sore_throat",
    "I have body pain", "body_pain",
    "I have nausea", "nausea",
    "I feel dizzy", "dizziness",
    "I have chest pain", "chest_pain",
    "I have shortness of breath", "shortness_breath",
    "I have diarrhea", "diarrhea",
    "I feel anxious", "anxiety",
    "I have vomiting", "vomiting",
    "I feel weak", "weakness",
    "I have skin rash", "skin_rash",
    "I have joint pain", "joint_pain",
    "I feel bloated", "bloating",
    "I have high blood pressure", "high_bp",
    "I have low blood pressure", "low_bp",
    "I have a rapid heartbeat", "palpitations",
    "I feel shivering", "shivering",
    "I have chills", "chills",
    "I have a stomach ache", "stomach_ache",
    "I feel constipated", "constipation",
    "I have a runny nose", "runny_nose",
    "I have ear pain", "ear_pain",
    "I have an eye infection", "eye_infection",
]

# Responses for each symptom
RESPONSES = {
    "fever": "A fever can indicate an infection. Rest, stay hydrated, and monitor your temperature.",
    "cold": "The common cold often includes sneezing, sore throat, and fatigue. Drink warm fluids and rest.",
    "headache": "Headaches can result from stress or dehydration. Take rest and drink water.",
    "cough": "Persistent cough can be caused by a cold or allergies. Try honey tea for relief.",
    "fatigue": "Fatigue may result from stress or lack of sleep. Aim for 7-8 hours of rest daily.",
    "sore_throat": "A sore throat can indicate a cold or infection. Gargle with warm salt water for relief.",
    "body_pain": "Body aches can result from flu or overexertion. Take a warm bath and rest.",
    "nausea": "Nausea can be caused by digestive issues. Try sipping ginger tea or clear fluids.",
    "dizziness": "Dizziness might result from low blood sugar or dehydration. Eat a light snack and drink water.",
    "chest_pain": "Chest pain may be serious. Please seek immediate medical attention.",
    "shortness_breath": "Shortness of breath can indicate respiratory issues. Rest and consult a doctor if it persists.",
    "diarrhea": "Diarrhea can result from infections or food issues. Stay hydrated with ORS.",
    "anxiety": "Anxiety can cause restlessness. Try breathing exercises or consult a therapist.",
    "vomiting": "Vomiting can be caused by indigestion. Stay hydrated and rest your stomach.",
    "weakness": "General weakness can result from illness. Eat nutritious food and get plenty of rest.",
    "skin_rash": "Skin rashes may indicate allergies. Avoid irritants and consult a dermatologist.",
    "joint_pain": "Joint pain may be due to arthritis or overuse. Apply warm compress and rest.",
    "bloating": "Bloating can result from indigestion. Avoid carbonated drinks and eat smaller meals.",
    "high_bp": "High blood pressure requires monitoring. Reduce salt intake and avoid stress.",
    "low_bp": "Low blood pressure may cause dizziness. Drink water and eat salty snacks.",
    "palpitations": "Rapid heartbeat can result from stress or caffeine. Relax and consult a doctor if it persists.",
    "shivering": "Shivering may accompany fever. Keep warm and monitor your temperature.",
    "chills": "Chills often occur with infections. Rest and drink warm fluids.",
    "stomach_ache": "Stomach pain can result from indigestion. Avoid heavy meals and try warm water.",
    "constipation": "Constipation can be relieved by eating fiber-rich foods and staying hydrated.",
    "runny_nose": "A runny nose is common in colds. Use steam inhalation for relief.",
    "ear_pain": "Ear pain can result from infections. Avoid cold air and consult a doctor if severe.",
    "eye_infection": "Eye infections require care. Avoid touching your eyes and consult an ophthalmologist.",
}

# Train the Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_data[::2])  # User input examples
y = training_data[1::2]  # Corresponding classes (labels)

model = MultinomialNB()
model.fit(X, y)

# Function to predict response
def get_response(user_input):
    """
    Predict the response for user input using the trained ML model.
    """
    user_input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vectorized)[0]  # Predict class
    return RESPONSES.get(prediction, "I'm sorry, I don't understand your symptoms. Please consult a doctor for a detailed diagnosis.")

# Home View
def home(request):
    """
    View to handle the chatbot interaction and display the response.
    """
    response = ""
    if request.method == "POST":
        user_input = request.POST.get('user_input')  # Get user input from the form
        response = get_response(user_input)
    return render(request, 'chatbot/home.html', {'response': response})
