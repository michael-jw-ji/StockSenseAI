# Stock Price Predictor

An AI-powered stock price prediction application that combines neural networks and sentiment analysis to provide comprehensive stock movement predictions.

## Features

- **Neural Network Analysis**: Uses MLPRegressor with technical indicators for price prediction
- **Sentiment Analysis**: Analyzes news sentiment using VADER sentiment analysis
- **Technical Indicators**: Includes moving averages, volume analysis, and price changes
- **Real-time Data**: Fetches live stock data from Yahoo Finance
- **News Integration**: Incorporates recent news sentiment for comprehensive analysis
- **Beautiful UI**: Modern, responsive React frontend with TypeScript
- **RESTful API**: Flask backend with CORS support

## Technology Stack

### Backend

- **Python 3.8+**
- **Flask**: Web framework
- **scikit-learn**: Machine learning library
- **yfinance**: Stock data fetching
- **NLTK**: Natural language processing
- **VADER Sentiment**: Sentiment analysis
- **pandas & numpy**: Data manipulation

### Frontend

- **React 18**: UI framework
- **TypeScript**: Type safety
- **CSS3**: Modern styling with gradients and animations
- **Axios**: HTTP client

## Project Structure

```
stock-price-predictor/
├── stock_predictor.py          # Main Flask backend with ML models
├── requirements.txt            # Python dependencies
├── package.json               # Node.js dependencies and scripts
├── package-lock.json          # Locked dependency versions
├── tsconfig.json              # TypeScript configuration
├── README.md                  # Project documentation
├── public/
│   └── index.html            # Main HTML template
└── src/
    ├── App.tsx               # Main React application
    ├── App.css               # App-level styles
    ├── index.tsx             # React entry point
    ├── index.css             # Global styles
    └── components/
        ├── Header.tsx        # Header component
        ├── Header.css        # Header styles
        ├── StockPredictor.tsx # Main stock prediction component
        └── StockPredictor.css # Stock predictor styles
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

### Quick Start

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd stock-price-predictor
   ```

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Node.js dependencies:**

   ```bash
   npm install
   ```

4. **Start the backend server:**

   ```bash
   python stock_predictor.py
   ```

   The server will start on `http://localhost:5000`

5. **Start the frontend (in a new terminal):**
   ```bash
   npm start
   ```
   The frontend will open on `http://localhost:3000`

## Usage

1. **Open the application** in your browser
2. **Enter a stock ticker symbol** (e.g., AAPL, MSFT, NVDA, TSLA)
3. **Click "Analyze Stock"** to analyze the stock
4. **View comprehensive results** including:
   - Current vs. predicted price
   - Buy/Sell/Hold recommendation
   - Confidence level
   - News sentiment analysis
   - Technical analysis reasoning

## API Endpoints

### POST /predict

Predicts stock movement for a given ticker.

**Request:**

```json
{
  "ticker": "AAPL"
}
```

**Response:**

```json
{
  "ticker": "AAPL",
  "current_price": 150.25,
  "predicted_price": 152.80,
  "predicted_change_percent": 1.70,
  "movement": "BUY",
  "confidence": "Medium",
  "reasoning": "Technical analysis and sentiment...",
  "sentiment_score": 0.15,
  "sentiment_analysis": "Positive - News sentiment is generally favorable",
  "analysis_date": "2025-01-15 14:30:00",
  "features_used": ["Open", "High", "Low", "Close", "Volume", "sentiment_score"],
  "recent_articles": [...]
}
```

### GET /health

Health check endpoint.

## Technical Details

### Neural Network Architecture

- **Type**: Multi-layer Perceptron Regressor
- **Layers**: 100 → 50 → 25 neurons
- **Activation**: ReLU
- **Optimizer**: Adam
- **Max Iterations**: 1000

### Features Used

1. Open, High, Low, Close prices
2. Volume data
3. Moving averages (20-day)
4. News sentiment scores

### Sentiment Analysis

- Uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Analyzes news headlines
- Aggregates daily sentiment scores
- Preprocesses text (removes stopwords, tokenizes)

## Data Sources

- **Stock Data**: Yahoo Finance API
- **News Data**: NewsAPI (requires API key)
- **Technical Indicators**: Calculated from price data

## Disclaimer

⚠️ **Important**: This application is for educational and research purposes only. Stock predictions are based on AI analysis and should not be considered as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.

## Troubleshooting

### Common Issues

#### 1. **Python dependencies missing**

**Symptoms**: `ModuleNotFoundError` or import errors
**Solutions**:

```bash
pip install -r requirements.txt
```

#### 2. **Node.js dependencies missing**

**Symptoms**: `Cannot find module` errors
**Solutions**:

```bash
npm install
```

#### 3. **Port already in use**

**Symptoms**: `Address already in use` errors
**Solutions**:

- Close other applications using ports 3000 or 5000
- Kill processes: `netstat -ano | findstr :5000` (Windows) or `lsof -i :5000` (Mac/Linux)

#### 4. **News API errors**

**Symptoms**: News sentiment analysis fails
**Solutions**:

- Get a free API key from [https://newsapi.org/](https://newsapi.org/)
- Update the API key in `stock_predictor.py`

### Performance Tips

- The neural network trains on each prediction for accuracy
- First prediction may take longer due to model initialization
- Subsequent predictions for the same stock are faster

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the API documentation
3. Open an issue on GitHub

