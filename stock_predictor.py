import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class StockPredictor:
    def __init__(self):
        self.is_trained = False
        
    def get_stock_data(self, ticker, days=30):
        try:
            today = datetime.today()
            start_date = (today - BDay(days)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            
            stock = yf.Ticker(ticker)
            stock_data = stock.history(start=start_date, end=end_date)
            stock_data.reset_index(inplace=True)
            
            if stock_data.empty:
                return None
                
            return stock_data
        except Exception as e:
            return None
    
    def get_news_data(self, ticker, days=28):
        try:
            api_key = "68a7a3c0ded4438e9277307a77011105"
            url = 'https://newsapi.org/v2/everything'
            
            params = {
                'q': ticker,
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'apiKey': api_key,
                'pageSize': 100,
                'language': 'en'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'ok':
                articles = data['articles']
                news_data = pd.DataFrame(articles)
                news_data = news_data[['publishedAt', 'title']]
                news_data.columns = ['date', 'headline']
                
                return news_data
            else:
                return None
                
        except Exception as e:
            return None
    
    def analyze_sentiment(self, news_data):
        if news_data is None or news_data.empty:
            return None
            
        try:
            stop_words = set(stopwords.words('english'))
            
            def preprocess_text(text):
                if pd.isna(text):
                    return ''
                words = word_tokenize(str(text))
                words = [word for word in words if word.isalpha()]
                words = [word for word in words if word.lower() not in stop_words]
                return ' '.join(words)
            
            news_data['cleaned_headline'] = news_data['headline'].apply(preprocess_text)
            
            analyzer = SentimentIntensityAnalyzer()
            
            def get_sentiment_score(text):
                score = analyzer.polarity_scores(text)
                return score['compound']
            
            news_data['sentiment_score'] = news_data['cleaned_headline'].apply(get_sentiment_score)
            
            news_data['date'] = pd.to_datetime(news_data['date']).dt.date
            aggregated_sentiment = news_data.groupby('date')['sentiment_score'].sum().reset_index()
            
            return aggregated_sentiment
            
        except Exception as e:
            return None
    
    def prepare_features(self, stock_data, sentiment_data):
        try:
            stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
            
            if sentiment_data is not None and not sentiment_data.empty:
                combined_data = pd.merge(stock_data, sentiment_data, left_on='Date', right_on='date', how='inner')
            else:
                combined_data = stock_data.copy()
                combined_data['sentiment_score'] = 0
            
            initial_rows = len(combined_data)
            combined_data = combined_data.dropna()
            final_rows = len(combined_data)
            
            available_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_score']
            feature_columns = [col for col in available_columns if col in combined_data.columns]
            
            features_df = combined_data[feature_columns].copy()
            features_df['target'] = combined_data['Close']
            
            return features_df
            
        except Exception as e:
            return None
    
    def train_model_with_data(self, X, y, model, scaler):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return model
            
        except Exception as e:
            return None
    
    def predict_stock_movement(self, ticker):
        try:
            import random
            random_seed = random.randint(1, 1000)
            
            fresh_model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=random_seed
            )
            fresh_scaler = StandardScaler()
            
            stock_data = self.get_stock_data(ticker)
            if stock_data is None or stock_data.empty:
                return None
            
            news_data = self.get_news_data(ticker)
            sentiment_data = self.analyze_sentiment(news_data)
            
            recent_articles = []
            if news_data is not None and not news_data.empty:
                try:
                    for i, row in news_data.head(3).iterrows():
                        recent_articles.append({
                            'title': row['headline'],
                            'description': row['headline'],
                            'url': '',
                            'publishedAt': str(row['date']),
                            'source': 'NewsAPI'
                        })
                except Exception as e:
                    recent_articles = []
            
            features_df = self.prepare_features(stock_data, sentiment_data)
            
            if features_df is None or features_df.empty:
                return None
            
            X = features_df.drop(['target'], axis=1)
            y = features_df['target']
            
            model = self.train_model_with_data(X, y, fresh_model, fresh_scaler)
            
            if model is None:
                return None
            
            latest_features = X.iloc[-1:].values
            latest_features_scaled = fresh_scaler.transform(latest_features)
            predicted_change = model.predict(latest_features_scaled)[0]
            
            current_price = stock_data['Close'].iloc[-1]
            predicted_change_percent = (predicted_change / current_price - 1) * 100
            predicted_price = current_price * (1 + predicted_change_percent / 100)
            
            if predicted_change_percent > 2:
                movement = "BUY"
                confidence = "High"
            elif predicted_change_percent > 0.5:
                movement = "BUY"
                confidence = "Medium"
            elif predicted_change_percent < -2:
                movement = "SELL"
                confidence = "High"
            elif predicted_change_percent < -0.5:
                movement = "SELL"
                confidence = "Medium"
            else:
                movement = "HOLD"
                confidence = "Medium"
            
            reasoning = self.generate_recommendation_reasoning(
                movement, predicted_change_percent, sentiment_data, stock_data
            )
            
            X_scaled = fresh_scaler.transform(X)
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            final_sentiment_score = 0
            if sentiment_data is not None and not sentiment_data.empty:
                final_sentiment_score = sentiment_data['sentiment_score'].iloc[-1]
            
            return {
                'ticker': ticker.upper(),
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'predicted_change_percent': round(predicted_change_percent, 2),
                'movement': movement,
                'confidence': confidence,
                'reasoning': reasoning,
                'sentiment_score': round(final_sentiment_score, 3),
                'sentiment_analysis': self.get_sentiment_description(final_sentiment_score),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'features_used': list(X.columns),
                'recent_articles': recent_articles
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def generate_recommendation_reasoning(self, movement, change_percent, sentiment_data, stock_data):
        sentiment_score = 0
        if sentiment_data is not None and not sentiment_data.empty:
            sentiment_score = sentiment_data['sentiment_score'].iloc[-1]
        
        current_price = stock_data['Close'].iloc[-1]
        
        reasoning_parts = []
        
        if current_price > stock_data['Close'].rolling(20).mean().iloc[-1] * 1.05:
            reasoning_parts.append("Stock is trading above its 20-day moving average, indicating upward momentum.")
        elif current_price < stock_data['Close'].rolling(20).mean().iloc[-1] * 0.95:
            reasoning_parts.append("Stock is trading below its 20-day moving average, suggesting downward pressure.")
        else:
            reasoning_parts.append("Stock is trading near its 20-day moving average, showing neutral momentum.")
        
        current_volume = stock_data['Volume'].iloc[-1]
        avg_volume = stock_data['Volume'].rolling(20).mean().iloc[-1]
        
        if current_volume > avg_volume * 1.5:
            reasoning_parts.append("Trading volume is significantly above average, confirming strong market interest.")
        elif current_volume < avg_volume * 0.5:
            reasoning_parts.append("Trading volume is below average, suggesting limited market participation.")
        else:
            reasoning_parts.append("Trading volume is at normal levels.")
        
        if sentiment_score > 0.3:
            reasoning_parts.append("News sentiment is strongly positive, which typically supports higher prices.")
        elif sentiment_score > 0.1:
            reasoning_parts.append("News sentiment is moderately positive, providing some upward support.")
        elif sentiment_score < -0.3:
            reasoning_parts.append("News sentiment is strongly negative, which typically pressures prices lower.")
        elif sentiment_score < -0.1:
            reasoning_parts.append("News sentiment is moderately negative, creating some downward pressure.")
        else:
            reasoning_parts.append("News sentiment is neutral, having minimal impact on price direction.")
        
        if movement == "BUY":
            if change_percent > 3:
                reasoning_parts.append("Technical indicators and sentiment analysis suggest strong upward potential.")
            else:
                reasoning_parts.append("Moderate upward movement expected based on current market conditions.")
        elif movement == "SELL":
            if change_percent < -3:
                reasoning_parts.append("Technical indicators and sentiment analysis suggest strong downward pressure.")
            else:
                reasoning_parts.append("Moderate downward movement expected based on current market conditions.")
        else:
            reasoning_parts.append("Mixed signals suggest the stock will likely trade sideways in the near term.")
        
        return " ".join(reasoning_parts)

    def get_sentiment_description(self, compound_score):
        if compound_score > 0.5:
            return "Very Positive - Strong bullish sentiment in recent news"
        elif compound_score > 0.1:
            return "Positive - Generally favorable news sentiment"
        elif compound_score < -0.5:
            return "Very Negative - Strong bearish sentiment in recent news"
        elif compound_score < -0.1:
            return "Negative - Generally unfavorable news sentiment"
        else:
            return "Neutral - Balanced news sentiment with no clear bias"

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

predictor = StockPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({"error": "Ticker symbol is required"}), 400
        
        result = predictor.predict_stock_movement(ticker)
        
        if result is None:
            return jsonify({"error": "Could not perform prediction"}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)