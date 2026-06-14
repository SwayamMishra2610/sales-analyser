# 🎉 Diwali Sales Analytics Dashboard

A comprehensive web-based analytics platform for analyzing Diwali sales data with advanced machine learning capabilities.

## ✨ Features

- **📊 Exploratory Data Analysis (EDA)**
  - Interactive charts and visualizations
  - Key metrics and statistics
  - Revenue breakdowns by category, state, segment
  - Correlation heatmaps
  - Data distribution analysis

- **🤖 Machine Learning Models**
  - **Spending Predictor**: Predict customer spending based on demographics
  - **Customer Segmentation**: K-Means clustering into 4 business segments
  - **Category Recommender**: Suggest products based on customer profile

- **💾 SQL Query Interface**
  - Write custom SQL queries
  - View results in interactive tables
  - Export data as CSV

- **📁 Data Management**
  - Upload CSV or Excel files
  - Automatic data validation
  - Support for different data formats


## 🚀 Live Demo

Visit the live app: [Diwali Sales Dashboard](https://your-username-diwali-sales-dashboard.streamlit.app)

(Update this link with your actual Streamlit Cloud URL after deployment)


## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML/Data**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Database**: SQLite, SQLAlchemy
- **Deployment**: Streamlit Cloud


## 📋 Requirements

```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
sqlalchemy==2.0.20
openpyxl==3.1.2
```


## 🏃 Quick Start

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/diwali-sales-dashboard.git
   cd diwali-sales-dashboard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run Appp.py
   ```

5. **Open in browser**
   ```
   http://localhost:8501
   ```


## 📁 Project Structure

```
diwali-sales-dashboard/
├── Appp.py                    # Main Streamlit application
├── ml_models.py               # Machine learning models
├── data_pipeline.py           # SQL data pipeline
├── Diwali_Sales_Data.csv      # Dataset
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```


## 📊 Dataset

The app uses the Diwali Sales Data with the following columns:

- **User_ID**: Customer identifier
- **Cust_name**: Customer name
- **Product_ID**: Product identifier
- **Gender**: Customer gender
- **Age Group**: Age bracket (0-17, 18-25, 26-35, etc.)
- **Age**: Exact age
- **Marital_Status**: Married (1) or Unmarried (0)
- **State**: Customer's state
- **Zone**: Geographic zone
- **Occupation**: Customer's occupation
- **Product_Category**: Type of product
- **Orders**: Number of orders
- **Amount**: Purchase amount in rupees


## 🤖 ML Models

### 1. Spending Prediction Regression
- **Algorithm**: Gradient Boosting Regressor
- **Features**: Age Group, Gender, Occupation, Zone
- **Target**: Amount (spending)
- **Performance**: ~60% R² score

### 2. Customer Segmentation
- **Algorithm**: K-Means Clustering (k=4)
- **Features**: Total Spend, Avg Ticket, Total Orders, Transactions
- **Segments**:
  - 🏆 High-Value Frequent Buyers
  - 🌟 Young Tech Enthusiasts
  - 💡 Value-Conscious Shoppers
  - 🎯 Occasional Premium Buyers

### 3. Category Recommender
- **Algorithm**: K-Nearest Neighbors (k=15)
- **Features**: Gender, Age Group, Occupation, Zone, Marital Status
- **Target**: Product Category
- **Accuracy**: ~35% (baseline for multiclass classification)


## 🎯 Usage

### Uploading Data
1. Click "Upload Data" in the sidebar
2. Select CSV or Excel file
3. App automatically validates and processes

### EDA Tab
1. View key metrics
2. Explore distributions
3. See correlations
4. Analyze segments

### ML Models Tab
1. Click "🚀 Train All Models"
2. Wait for training (15-20 seconds)
3. View model performance
4. Make predictions
5. Look up customer segments
6. Get recommendations

### SQL Query Tab
1. Write SQL query
2. Click "Run Query"
3. View results
4. Export as CSV


## 🔧 Configuration

Model hyperparameters can be adjusted in `ml_models.py`:

```python
# Regression model
GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    random_state=42,
)

# Clustering
KMeans(n_clusters=4, random_state=42, n_init=10)

# Recommender
KNeighborsClassifier(n_neighbors=15, weights="distance")
```


## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Click "New app"
4. Select repository and main file
5. Deploy!

See `DEPLOYMENT_GUIDE.md` for other deployment options.


## 📈 Performance

- **Data Loading**: < 1 second
- **EDA Rendering**: < 2 seconds
- **Model Training**: 15-20 seconds
- **Predictions**: < 1 second
- **Recommendations**: < 1 second


## 🐛 Troubleshooting

### "ml_models.py not found"
→ Make sure ml_models.py is in the same folder as Appp.py

### "CSV file not found"
→ Ensure Diwali_Sales_Data.csv is in the project folder

### "ModuleNotFoundError"
→ Run: `pip install -r requirements.txt`

### ValueError with NaN
→ Update to latest ml_models.py (has NaN handling)

### Slow performance
→ This is normal for first load; subsequent loads are faster


## 📝 Notes

- The app uses SQLite for data storage
- ML models are trained in-memory
- Maximum file upload size: 200 MB (Streamlit Cloud limit)
- Free tier has resource limitations


## 🤝 Contributing

Feel free to fork, modify, and enhance the dashboard!

Suggestions for improvements:
- Add more ML models
- Enhance visualizations
- Add forecasting capabilities
- Include more data sources


## 📄 License

This project is open source and available under the MIT License.


## 👤 Author

Created for Diwali Sales Analytics

For questions or support, please create an issue on GitHub.


## 🙏 Acknowledgments

- Streamlit for the amazing framework
- scikit-learn for ML capabilities
- pandas for data manipulation


---

**Last Updated**: June 2024  
**Status**: ✅ Production Ready  
**Version**: 1.0.0
