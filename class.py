import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    return scores['compound']

def aspect_sentiment(text, aspect):
    pass

def main():
    text_data = [
        "I love this product! It's amazing.",
        "The customer service was terrible, but the product quality is good.",
        "The user interface is intuitive and easy to use.",
    ]

    preprocessed_data = [preprocess_text(text) for text in text_data]

    sentiment_scores = [analyze_sentiment(' '.join(tokens)) for tokens in preprocessed_data]

    df = pd.DataFrame({'Text': text_data, 'Sentiment Score': sentiment_scores})
    plt.bar(df['Text'], df['Sentiment Score'])
    plt.xlabel('Text')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis Results')
    plt.xticks(rotation=45, ha='right')
    plt.show()

if __name__ == "__main__":
    main()
