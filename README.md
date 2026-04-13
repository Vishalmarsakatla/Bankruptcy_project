# 📊 Bankruptcy Prediction System - Streamlit App

A comprehensive machine learning web application for predicting bankruptcy risk using multiple classification models.

## 🌟 Features

- **Data Upload & Overview**: Upload Excel files and view data statistics
- **Exploratory Data Analysis**: Interactive visualizations and insights
- **6 ML Models**: 
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - K-Nearest Neighbors
  - Support Vector Machine
  - Naive Bayes
- **Model Comparison**: Compare performance across all models
- **Live Predictions**: Make real-time bankruptcy predictions
- **Professional UI**: Clean, intuitive interface built with Streamlit

## 📋 Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## 🚀 Quick Start

### Option 1: Local Deployment

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the app**:
```bash
streamlit run streamlit_app.py
```

3. **Open in browser**:
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in terminal

### Option 2: Using the Run Script

**On Windows**:
```bash
./run.bat
```

**On Mac/Linux**:
```bash
chmod +x run.sh
./run.sh
```

## 📁 File Structure

```
bankruptcy-prediction/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── run.sh                     # Linux/Mac startup script
└── run.bat                    # Windows startup script
```

## 💡 Usage Guide

### 1. Upload Data
- Click on "Browse files" in the sidebar
- Select your bankruptcy Excel file
- Data will be automatically processed

### 2. Explore Pages

**🏠 Home**: Overview and instructions

**📊 Data Overview**: 
- View dataset statistics
- Check data quality
- Preview data samples

**📈 EDA**:
- Target variable distribution
- Feature distributions
- Correlation analysis
- Feature vs. target relationships

**🤖 Model Training**:
- Train all 6 models automatically
- View confusion matrices
- Check classification reports
- Analyze ROC curves

**🔮 Make Prediction**:
- Select a trained model
- Adjust risk factor sliders
- Get instant bankruptcy prediction
- View probability scores

**📉 Model Comparison**:
- Compare all model performances
- View accuracy charts
- Get best model recommendation

## 🌐 Cloud Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo>
git push -u origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file path: `streamlit_app.py`
7. Click "Deploy"

### Heroku Deployment

1. Create `Procfile`:
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Railway Deployment

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Add environment variables if needed
5. Railway will auto-detect Streamlit and deploy

### Render Deployment

1. Go to [render.com](https://render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
6. Click "Create Web Service"

## 📊 Sample Data Format

Your Excel file should have data in the first column, semicolon-separated:

```
industrial_risk;management_risk;financial_flexibility;credibility;competitiveness;operating_risk;class
0.5;1;0;0;0;0.5;bankruptcy
0;1;0;0;0;1;bankruptcy
```

## 🛠️ Customization

### Adding New Models

Edit `streamlit_app.py` and add to the `models` dictionary in `train_models()`:

```python
models = {
    'Your Model Name': (YourModelClass(), use_scaling_bool),
    # ... existing models
}
```

### Changing Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

### Issue: "File upload not working"
**Solution**: Check file format (must be .xlsx or .xls)

### Issue: "Models not training"
**Solution**: Ensure data is uploaded and navigate to "Model Training" page

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Created for bankruptcy prediction analysis and deployment.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

---

**Need Help?** Open an issue or contact the development team.
