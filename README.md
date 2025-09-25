# Stock Market Trend Predictor using ML & DL

A comprehensive web application that predicts stock market trends using Machine Learning and Deep Learning algorithms. The application provides real-time stock analysis, technical indicators, and trend predictions with an intuitive user interface.

## Features

### Core Functionality
- **Multi-Model Prediction**: Combines Random Forest, SVM, XGBoost, and LSTM models for accurate predictions
- **Real-time Data**: Fetches live stock data using Yahoo Finance API
- **Technical Analysis**: Includes RSI, MACD, SMA, EMA, and other technical indicators
- **Interactive Charts**: Visual representation of stock prices with technical indicators
- **Watchlist Management**: Personal watchlist for tracking multiple stocks
- **User Authentication**: Secure login system for remote users and service providers

### Technical Indicators
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- On-Balance Volume (OBV)
- Williams %R
- Support and Resistance levels

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: scikit-learn, XGBoost
- **Deep Learning**: TensorFlow/Keras
- **Data Source**: Yahoo Finance (yfinance)
- **Visualization**: Matplotlib
- **Data Processing**: Pandas, NumPy

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/KIRANRW9/stock-trend-predictor.git
cd stock-trend-predictor
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the application**
Open your web browser and navigate to `http://localhost:8501`

## Requirements

```
streamlit
pandas
numpy
yfinance
matplotlib
scikit-learn
xgboost
tensorflow
```

## Application Screenshots

### User Management

![User Registration](https://github.com/KIRANRW9/stock-trend-predictor/blob/main/Images/user-registration-form.png)
*User registration form for remote users*

![Remote User Login](https://github.com/KIRANRW9/stock-trend-predictor/blob/main/Images/remote-user-login.png)
*Login interface for remote users*

![Service Provider Registration](https://github.com/KIRANRW9/stock-trend-predictor/blob/main/Images/service-provider-registration.png)
*Registration form for service providers*

![Service Provider Login](https://github.com/KIRANRW9/stock-trend-predictor/blob/main/Images/service-provider-login.png)
*Service provider login interface*

![Service Provider Dashboard](https://github.com/KIRANRW9/stock-trend-predictor/blob/main/Images/service-provider-dashboard.png)
*Admin dashboard for managing registered users*

![User Dashboard](https://github.com/KIRANRW9/stock-trend-predictor/blob/main/Images/user-dashboard-after-login.png)
*User dashboard after successful login*

### User Interface

![Home Dashboard](https://github.com/KIRANRW9/stock-trend-predictor/blob/main/Images/home-dashboard-add-stock.png)
*Home dashboard with watchlist functionality*

![Watchlist Analysis](https://github.com/KIRANRW9/stock-trend-predictor/blob/main/Images/watchlist-analysis-results.png)
*Watchlist showing uptrend and downtrend predictions*

![Individual Stock Prediction](https://github.com/KIRANRW9/stock-trend-predictor/blob/main/Images/individual-stock-prediction.png)
*Individual stock analysis and prediction interface*

![Stock Prediction Results](https://github.com/KIRANRW9/stock-trend-predictor/blob/main/Images/stock-prediction-results.png)
*Detailed stock analysis with trading metrics and prediction results*

### Technical Analysis

![Technical Indicators Chart](https://github.com/KIRANRW9/stock-trend-predictor/blob/main/Images/technical-indicators-chart.png)
*WIPRO.NS comprehensive stock analysis displaying price movements from October 2024 to September 2025 with SMA (orange) and EMA (red) moving averages, support level at ₹241.27 (green dashed line), resistance level at ₹259.80 (red dashed line), and RSI momentum oscillator below showing overbought (70) and oversold (30) threshold levels. Green upward arrow indicates ML model's bullish prediction for next trading session.*

## How to Use

### For Remote Users

1. **Registration & Login**
   - Register as a new user or login with existing credentials
   - Access personalized dashboard

2. **Watchlist Management**
   - Add stocks to your personal watchlist
   - Format: Use `.NS` suffix for NSE stocks (e.g., `RELIANCE.NS`, `TCS.NS`)
   - Analyze entire watchlist with one click

3. **Individual Stock Analysis**
   - Search for any stock symbol
   - Get detailed prediction with technical analysis
   - View comprehensive trading metrics

### For Service Providers

1. **Admin Access**
   - Register as a service provider
   - Access admin dashboard
   - Monitor all registered users

## Machine Learning Models

The application employs an ensemble approach combining multiple algorithms:

### Traditional ML Models
- **Random Forest**: Handles non-linear patterns and feature importance
- **Support Vector Machine (SVM)**: Effective for classification in high-dimensional space
- **XGBoost**: Gradient boosting for enhanced accuracy

### Deep Learning Model
- **LSTM (Long Short-Term Memory)**: Captures temporal dependencies in stock price sequences

### Prediction Logic
- Each model provides individual predictions
- Final prediction uses weighted ensemble voting
- Confidence scores based on model accuracies

## Technical Features

### Data Processing
- **Real-time Data Fetching**: Yahoo Finance API integration
- **Feature Engineering**: 9 technical indicators
- **Data Validation**: Handles missing values and data quality issues
- **Normalization**: StandardScaler for feature scaling

### Performance Optimization
- **Caching**: Streamlit cache for improved performance
- **Error Handling**: Comprehensive exception handling
- **Data Persistence**: JSON-based user data storage

## Supported Stock Markets

- **Indian Stock Market (NSE)**: Use `.NS` suffix
  - Examples: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- **Indian Stock Market (BSE)**: Use `.BO` suffix
  - Examples: `RELIANCE.BO`, `TCS.BO`
- **US Stock Market**: Direct ticker symbols
  - Examples: `AAPL`, `GOOGL`, `MSFT`

## Important Notes

### Risk Disclaimer
- This application is for educational and research purposes only
- Stock market predictions are inherently uncertain
- Past performance does not guarantee future results
- Always consult with financial advisors before making investment decisions
- Users should conduct their own research before trading

### Data Limitations
- Predictions based on historical data and technical indicators
- Market sentiment and external factors not considered
- Real-time data subject to API limitations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comments for complex logic
- Test thoroughly before submitting
- Update documentation as needed

## About the Developer

**Kiran Rangu - Data Analyst | Business Intelligence | AI & Data Science Graduate**

- AI & Data Science Graduate with strong foundation in statistical analysis and machine learning concepts
- BI Enthusiast skilled in Power BI, DAX, Python, SQL, and learning advanced data modeling techniques
- Analytical Thinker passionate about transforming datasets into meaningful business insights
- Dashboard Creator focused on building clear, actionable visualizations and KPI tracking
- Business-Minded professional eager to identify growth opportunities and performance improvements
- Quick Learner with hands-on project experience in analytics and continuous skill development
- Self-Driven Innovator with hands-on project portfolio and relentless pursuit of cutting-edge analytics skills

**Connect with Me:**
- **LinkedIn**: [linkedin.com/in/kiranrangu](https://linkedin.com/in/kiranrangu)
- **GitHub**: [github.com/KIRANRW9](https://github.com/KIRANRW9)

## Contact & Support

**For Project-Related Inquiries:**
- **Email**: kiranrw09@gmail.com
- **LinkedIn**: [linkedin.com/in/kiranrangu](https://linkedin.com/in/kiranrangu)
- **Issues**: Please use [GitHub Issues](https://github.com/KIRANRW9/stock-trend-predictor/issues) for bug reports and feature requests

**For Technical Support:**
1. Check the [Issues](https://github.com/KIRANRW9/stock-trend-predictor/issues) section
2. Create a new issue with detailed description
3. Provide error logs and system information
4. Contact the developer for consultation on analytics strategy

**Note**: This application is created for educational and portfolio demonstration purposes. It showcases machine learning and data science skills using real market data. Please ensure proper risk management and consult financial advisors before making investment decisions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Yahoo Finance for providing stock data API
- Streamlit for the web application framework
- TensorFlow and scikit-learn communities for ML libraries
- Open source contributors and maintainers

## Tags

`Python` `Machine-Learning` `Deep-Learning` `Stock-Prediction` `Streamlit` `TensorFlow` `scikit-learn` `XGBoost` `LSTM` `Technical-Analysis` `Data-Science` `Portfolio-Project` `Kiran-Rangu` `AI` `Financial-Analytics`

---

**Made with ❤️ by [Kiran Rangu](https://github.com/KIRANRW9)**
