# Rewrite stock_predictor.py to match notebook exactly
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

# Download required NLTK data
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
        """Fetch stock data from Yahoo Finance following notebook approach"""
        try:
            # Use business days like in notebook
            today = datetime.today()
            start_date = (today - BDay(days)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')  # Include today's data
            
            stock = yf.Ticker(ticker)
            stock_data = stock.history(start=start_date, end=end_date)
            stock_data.reset_index(inplace=True)
            
            if stock_data.empty:
                return None
                
            return stock_data
        except Exception as e:
            return None
    
    def get_news_data(self, ticker, days=28):
        """Fetch news data for sentiment analysis following notebook approach"""
        try:
            # Use same API key and approach as notebook
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
                # Use same columns as notebook
                news_data = news_data[['publishedAt', 'title']]
                news_data.columns = ['date', 'headline']
                
                return news_data
            else:
                return None
                
        except Exception as e:
            return None
    
    def analyze_sentiment(self, news_data):
        """Analyze sentiment of news articles following notebook approach"""
        if news_data is None or news_data.empty:
            return None
            
        try:
            # Use same preprocessing as notebook
            stop_words = set(stopwords.words('english'))
            
            def preprocess_text(text):
                if pd.isna(text):
                    return ''
                words = word_tokenize(str(text))
                words = [word for word in words if word.isalpha()]
                words = [word for word in words if word.lower() not in stop_words]
                return ' '.join(words)
            
            # Apply same preprocessing as notebook
            news_data['cleaned_headline'] = news_data['headline'].apply(preprocess_text)
            
            # Use same sentiment analyzer as notebook
            analyzer = SentimentIntensityAnalyzer()
            
            def get_sentiment_score(text):
                score = analyzer.polarity_scores(text)
                return score['compound']
            
            news_data['sentiment_score'] = news_data['cleaned_headline'].apply(get_sentiment_score)
            
            # Aggregate sentiment by date like in notebook
            news_data['date'] = pd.to_datetime(news_data['date']).dt.date
            aggregated_sentiment = news_data.groupby('date')['sentiment_score'].sum().reset_index()
            
            return aggregated_sentiment
            
        except Exception as e:
            return None
    
    def prepare_features(self, stock_data, sentiment_data):
        """Prepare features for the neural network following notebook approach"""
        try:
            # Convert dates to same format as notebook
            stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
            
            # Merge with sentiment data like in notebook
            if sentiment_data is not None and not sentiment_data.empty:
                # Use inner merge like in notebook
                combined_data = pd.merge(stock_data, sentiment_data, left_on='Date', right_on='date', how='inner')
            else:
                combined_data = stock_data.copy()
                combined_data['sentiment_score'] = 0
            
            # Remove rows with NaN values
            initial_rows = len(combined_data)
            combined_data = combined_data.dropna()
            final_rows = len(combined_data)
            
            # Select features based on available columns
            available_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_score']
            feature_columns = [col for col in available_columns if col in combined_data.columns]
            
            # Create features DataFrame
            features_df = combined_data[feature_columns].copy()
            features_df['target'] = combined_data['Close']
            
            return features_df
            
        except Exception as e:
            return None
    
    def train_model_with_data(self, X, y, model, scaler):
        """Train a fresh model instance with provided data"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features using the fresh scaler
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train the fresh model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return model
            
        except Exception as e:
            return None
    
    def predict_stock_movement(self, ticker):
        """Predict stock movement using neural network and sentiment analysis following notebook approach"""
        try:
            # Create a fresh model instance for each stock to ensure independent analysis
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
            
            # Get stock data using notebook approach
            stock_data = self.get_stock_data(ticker)
            if stock_data is None or stock_data.empty:
                return None
            
            # Get news data and sentiment using notebook approach
            news_data = self.get_news_data(ticker)
            sentiment_data = self.analyze_sentiment(news_data)
            
            # Get first 3 news articles for display
            recent_articles = []
            if news_data is not None and not news_data.empty:
                try:
                    # Get the first 3 articles from the news data we already have
                    for i, row in news_data.head(3).iterrows():
                        recent_articles.append({
                            'title': row['headline'],
                            'description': row['headline'],  # Use headline as description since notebook only has title
                            'url': '',  # Notebook doesn't have URLs
                            'publishedAt': str(row['date']),
                            'source': 'NewsAPI'
                        })
                except Exception as e:
                    recent_articles = []
            
            # Prepare features using notebook approach
            features_df = self.prepare_features(stock_data, sentiment_data)
            
            if features_df is None or features_df.empty:
                return None
            
            # Prepare training data
            X = features_df.drop(['target'], axis=1)
            y = features_df['target']
            
            # Train fresh model for this specific stock
            model = self.train_model_with_data(X, y, fresh_model, fresh_scaler)
            
            if model is None:
                return None
            
            # Make prediction using the fresh model
            latest_features = X.iloc[-1:].values
            latest_features_scaled = fresh_scaler.transform(latest_features)
            predicted_change = model.predict(latest_features_scaled)[0]
            
            # Get current price - FIX THIS PART
            current_price = stock_data['Close'].iloc[-1]
            predicted_change_percent = (predicted_change / current_price - 1) * 100
            predicted_price = current_price * (1 + predicted_change_percent / 100)
            
            # Determine movement and confidence
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
            
            # Generate detailed reasoning
            reasoning = self.generate_recommendation_reasoning(
                movement, predicted_change_percent, sentiment_data, stock_data
            )
            
            # Calculate model performance
            X_scaled = fresh_scaler.transform(X)
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            # Calculate final sentiment score for response
            final_sentiment_score = 0
            if sentiment_data is not None and not sentiment_data.empty:
                # Get the most recent sentiment score
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
        """Generate detailed reasoning for the recommendation"""
        sentiment_score = 0
        if sentiment_data is not None and not sentiment_data.empty:
            sentiment_score = sentiment_data['sentiment_score'].iloc[-1]
        
        # Technical analysis reasoning
        current_price = stock_data['Close'].iloc[-1]
        
        reasoning_parts = []
        
        # Price analysis
        if current_price > stock_data['Close'].rolling(20).mean().iloc[-1] * 1.05:
            reasoning_parts.append("Stock is trading above its 20-day moving average, indicating upward momentum.")
        elif current_price < stock_data['Close'].rolling(20).mean().iloc[-1] * 0.95:
            reasoning_parts.append("Stock is trading below its 20-day moving average, suggesting downward pressure.")
        else:
            reasoning_parts.append("Stock is trading near its 20-day moving average, showing neutral momentum.")
        
        # Volume Analysis
        current_volume = stock_data['Volume'].iloc[-1]
        avg_volume = stock_data['Volume'].rolling(20).mean().iloc[-1]
        
        if current_volume > avg_volume * 1.5:
            reasoning_parts.append("Trading volume is significantly above average, confirming strong market interest.")
        elif current_volume < avg_volume * 0.5:
            reasoning_parts.append("Trading volume is below average, suggesting limited market participation.")
        else:
            reasoning_parts.append("Trading volume is at normal levels.")
        
        # Sentiment Analysis
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
        
        # Movement-specific reasoning
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
        else:  # HOLD
            reasoning_parts.append("Mixed signals suggest the stock will likely trade sideways in the near term.")
        
        return " ".join(reasoning_parts)

    def get_sentiment_description(self, compound_score):
        """Get human-readable sentiment description"""
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

# Flask app setup
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