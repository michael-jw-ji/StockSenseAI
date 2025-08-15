import React, { useState } from "react";
import axios from "axios";
import "./StockPredictor.css";

interface PredictionResult {
  ticker: string;
  current_price: number;
  predicted_price: number;
  predicted_change_percent: number;
  movement: string;
  confidence: string;
  reasoning: string;
  sentiment_score: number;
  sentiment_analysis: string;
  analysis_date: string;
  features_used: string[];
  recent_articles: Array<{
    title: string;
    description: string;
    url: string;
    publishedAt: string;
    source: string;
  }>;
}

const StockPredictor: React.FC = () => {
  const [ticker, setTicker] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string>("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!ticker.trim()) {
      setError("Please enter a stock ticker symbol");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await axios.post("http://localhost:5000/predict", {
        ticker: ticker.trim().toUpperCase(),
      });

      setResult(response.data);
    } catch (err: any) {
      if (err.response?.data?.error) {
        setError(err.response.data.error);
      } else {
        setError("Failed to get prediction. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const formatMovement = (movement: string) => {
    // Remove "STRONG" from movement text
    return movement.replace("STRONG ", "");
  };

  return (
    <div className="stock-predictor">
      <div className="predictor-container">
        {/* Left Column - Input Section */}
        <div className="input-section">
          <h2>Stock Analysis</h2>
          <p>
            Enter a stock ticker symbol to get AI-powered predictions using
            advanced neural networks and sentiment analysis
          </p>
          <form onSubmit={handleSubmit} className="prediction-form">
            <div className="input-group">
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                placeholder="Enter stock ticker (e.g., AAPL, MSFT, NVDA)"
                className="ticker-input"
                disabled={isLoading}
              />
              <button
                type="submit"
                className="predict-button"
                disabled={isLoading || !ticker.trim()}
              >
                {isLoading ? "Analyzing..." : "Analyze Stock"}
              </button>
            </div>
          </form>

          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          <div className="disclaimer">
            <h3>⚠️ Investment Disclaimer</h3>
            <p>
              This application is for educational and research purposes only.
              Stock predictions are based on AI analysis and should not be
              considered as financial advice. Past performance does not
              guarantee future results. Always do your own research and consult
              with financial professionals before making investment decisions.
            </p>
          </div>
        </div>

        {/* Center Column - Results */}
        {isLoading && (
          <div className="loading-section">
            <div className="loading-spinner"></div>
            <p>Analyzing {ticker.toUpperCase()}...</p>
            <p className="loading-details">
              Processing market data, analyzing news sentiment, and training
              neural network models...
            </p>
          </div>
        )}

        {result && (
          <div className="results-section">
            <div className="result-header">
              <h2>Analysis Results for {result.ticker}</h2>
              <div className="header-info">
                <span className="analysis-date">
                  Analysis Date: {result.analysis_date}
                </span>
                <span className="trading-date-badge">
                  📅 Data from latest trading session
                </span>
              </div>
            </div>

            <div className="prediction-summary">
              <div className="summary-card current-price">
                <h3>Current Market Price</h3>
                <p className="price-value">${result.current_price}</p>
              </div>

              <div className="summary-card predicted-price">
                <h3>AI Predicted Price</h3>
                <p className="price-value">${result.predicted_price}</p>
              </div>

              <div className="summary-card change-percent">
                <h3>Expected Movement</h3>
                <p
                  className={`price-value ${
                    result.predicted_change_percent >= 0
                      ? "positive"
                      : "negative"
                  }`}
                >
                  {result.predicted_change_percent >= 0 ? "+" : ""}
                  {result.predicted_change_percent}%
                </p>
              </div>
            </div>

            <div className="recommendation-section">
              <div className="recommendation-card">
                <h3>Investment Recommendation</h3>
                <div className="recommendation-content">
                  <span className="movement-badge">
                    {formatMovement(result.movement)}
                  </span>
                  <span className="confidence-badge">
                    {result.confidence} Confidence
                  </span>
                </div>
              </div>

              <div className="reasoning-section">
                <h3>Market Analysis & Reasoning</h3>
                <div className="reasoning-content">
                  <p>{result.reasoning}</p>
                </div>
              </div>
            </div>

            <div className="sentiment-card">
              <h3>News Sentiment Analysis</h3>
              <div className="sentiment-content">
                <div className="sentiment-score">
                  <span>Sentiment Score: </span>
                  <span
                    className={`score ${
                      result.sentiment_score >= 0 ? "positive" : "negative"
                    }`}
                  >
                    {result.sentiment_score >= 0 ? "+" : ""}
                    {result.sentiment_score}
                  </span>
                </div>
                <p className="sentiment-analysis">
                  {result.sentiment_analysis}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Right Column - Recent News */}
        <div className="right-sidebar">
          <h3>Recent News</h3>

          {result ? (
            <div className="news-section">
              {result.recent_articles.length === 0 ? (
                <p className="no-news-message">
                  No recent news available for {result.ticker}
                </p>
              ) : (
                <ul>
                  {result.recent_articles.map((article, index) => (
                    <li key={index}>
                      <a
                        href={article.url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {article.title}
                      </a>
                      <p>{article.description}</p>
                      <p>
                        Published on{" "}
                        {new Date(article.publishedAt).toLocaleDateString()}
                      </p>
                      <p>Source: {article.source}</p>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ) : (
            <div className="news-section">
              <p className="no-news-message">
                Enter a stock ticker to see recent news
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StockPredictor;
