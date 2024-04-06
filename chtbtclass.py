import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens

intents = {
    "greeting": {"patterns": ["hello", "hi", "hey"], "responses": ["Hello!", "Hi there!", "Hey!"]},
    "goodbye": {"patterns": ["bye", "see you later", "goodbye"], "responses": ["Goodbye!", "See you later!", "Take care!"]},
}

training_data = []
for intent, data in intents.items():
    for pattern in data['patterns']:
        tokens = preprocess_text(pattern)
        training_data.append((' '.join(tokens), intent))

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform([data[0] for data in training_data])
y_train = [data[1] for data in training_data]

def predict_intent(user_input):
    input_tokens = preprocess_text(user_input)
    input_vector = vectorizer.transform([' '.join(input_tokens)])
    similarity_scores = cosine_similarity(input_vector, X_train)
    max_sim_index = np.argmax(similarity_scores)
    return y_train[max_sim_index]

def get_response(intent):
    return np.random.choice(intents[intent]['responses'])

print("Chatbot: Hello! How can I help you?")
while True:
    user_input = input("You: ")
    intent = predict_intent(user_input)
    response = get_response(intent)
    print("Chatbot:", response)
