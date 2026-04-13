import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

# Page Configuration
st.set_page_config(
    page_title="Bankruptcy Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess bankruptcy data"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        
        # Split the semicolon-separated first column
        df = df.iloc[:, 0].astype(str).str.split(';', expand=True)
        
        # Set column names (skip the first row which is the header)
        df.columns = df.iloc[0].str.strip()
        df = df[1:].reset_index(drop=True)
        
        # Convert to numeric
        numeric_cols = ['industrial_risk', 'management_risk', 'financial_flexibility',
                       'credibility', 'competitiveness', 'operating_risk']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Encode target variable
        df['class'] = df['class'].map({
            'bankruptcy': 1,
            'non-bankruptcy': 0
        })
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def plot_to_streamlit(fig):
    """Convert matplotlib figure to streamlit"""
    st.pyplot(fig)
    plt.close()

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    """Train all models and return results"""
    results = {}
    
    # Scale data for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000), True),
        'Random Forest': (RandomForestClassifier(random_state=42), False),
        'Decision Tree': (DecisionTreeClassifier(random_state=42), False),
        'KNN': (KNeighborsClassifier(n_neighbors=5), True),
        'SVM': (SVC(probability=True, random_state=42), True),
        'Naive Bayes': (GaussianNB(), False)
    }
    
    for name, (model, use_scaling) in models.items():
        # Select appropriate data
        X_tr = X_train_scaled if use_scaling else X_train
        X_te = X_test_scaled if use_scaling else X_test
        
        # Train
        model.fit(X_tr, y_train)
        
        # Predict
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Metrics
        results[name] = {
            'model': model,
            'scaler': scaler if use_scaling else None,
            'use_scaling': use_scaling,
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    return results

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">📊 Bankruptcy Prediction System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "📊 Data Overview", "📈 EDA", "🤖 Model Training", "🔮 Make Prediction", "📉 Model Comparison"]
    )
    
    # File Upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload bankruptcy data (Excel)",
        type=['xlsx', 'xls'],
        help="Upload the bankruptcy prevention Excel file"
    )
    
    # Load data if uploaded
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.sidebar.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Store in session state
            if 'df' not in st.session_state or st.session_state.df is None:
                st.session_state.df = df
                
                # Prepare train-test split
                X = df.drop('class', axis=1)
                y = df['class']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
    else:
        st.session_state.df = None
    
    # Page Routing
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Overview":
        if st.session_state.get('df') is not None:
            show_data_overview(st.session_state.df)
        else:
            st.warning("⚠️ Please upload data first!")
    elif page == "📈 EDA":
        if st.session_state.get('df') is not None:
            show_eda(st.session_state.df)
        else:
            st.warning("⚠️ Please upload data first!")
    elif page == "🤖 Model Training":
        if st.session_state.get('df') is not None:
            show_model_training()
        else:
            st.warning("⚠️ Please upload data first!")
    elif page == "🔮 Make Prediction":
        if st.session_state.get('df') is not None:
            show_prediction()
        else:
            st.warning("⚠️ Please upload data first!")
    elif page == "📉 Model Comparison":
        if st.session_state.get('df') is not None:
            show_comparison()
        else:
            st.warning("⚠️ Please upload data first!")

def show_home():
    st.markdown("## Welcome to the Bankruptcy Prediction System! 🎯")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        ### 📊 Data Analysis
        - Upload your bankruptcy data
        - Explore distributions
        - Visualize patterns
        """)
    
    with col2:
        st.success("""
        ### 🤖 ML Models
        - 6 Classification Models
        - Performance Comparison
        - Cross-Validation
        """)
    
    with col3:
        st.warning("""
        ### 🔮 Predictions
        - Real-time predictions
        - Risk assessment
        - Interactive interface
        """)
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. **Upload Data**: Use the sidebar to upload your bankruptcy prevention Excel file
    2. **Explore**: Navigate through different pages to analyze the data
    3. **Train Models**: Train multiple ML models and compare performance
    4. **Predict**: Use the trained models to make bankruptcy predictions
    """)
    
    st.markdown("---")
    st.markdown("### 🎓 Models Included")
    models_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree', 'KNN', 'SVM', 'Naive Bayes'],
        'Type': ['Linear', 'Ensemble', 'Tree', 'Instance-based', 'Kernel', 'Probabilistic'],
        'Best For': ['Binary classification', 'High accuracy', 'Interpretability', 'Non-linear', 'Small datasets', 'Fast training']
    })
    st.dataframe(models_df, use_container_width=True)

def show_data_overview(df):
    st.markdown("## 📊 Data Overview")
    
    # Basic Info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Bankruptcy Cases", df[df['class'] == 1].shape[0])
    with col4:
        st.metric("Non-Bankruptcy Cases", df[df['class'] == 0].shape[0])
    
    st.markdown("---")
    
    # Data Preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistics
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Missing Values
    st.subheader("🔍 Data Quality")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values:**")
        missing = df.isnull().sum()
        st.dataframe(missing, use_container_width=True)
    
    with col2:
        st.write("**Data Types:**")
        dtypes = pd.DataFrame(df.dtypes, columns=['Type'])
        st.dataframe(dtypes, use_container_width=True)

def show_eda(df):
    st.markdown("## 📈 Exploratory Data Analysis")
    
    # Target Distribution
    st.subheader("🎯 Target Variable Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        df['class'].value_counts().plot(kind='bar', color=['#66c2a5', '#fc8d62'], ax=ax)
        plt.title("Class Distribution", fontsize=14, fontweight='bold')
        plt.xlabel("Class (0 = Non-Bankruptcy, 1 = Bankruptcy)")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plot_to_streamlit(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#e79647', '#78ebfa']
        df['class'].value_counts().plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            ax=ax
        )
        plt.title("Bankruptcy Distribution", fontsize=14, fontweight='bold')
        plt.ylabel('')
        plot_to_streamlit(fig)
    
    # Feature Distributions
    st.subheader("📊 Feature Distributions")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(df.columns[:-1]):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f"Distribution of {col}")
    
    plt.tight_layout()
    plot_to_streamlit(fig)
    
    # Feature vs Target
    st.subheader("🔍 Features vs Bankruptcy")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(df.columns[:-1]):
        sns.countplot(x=col, hue='class', data=df, palette='Set2', ax=axes[i])
        axes[i].set_title(f"{col} vs Bankruptcy")
        axes[i].legend(title='Class', labels=['Non-Bankruptcy', 'Bankruptcy'])
    
    plt.tight_layout()
    plot_to_streamlit(fig)
    
    # Correlation Heatmap
    st.subheader("🌡️ Correlation Matrix")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plot_to_streamlit(fig)

def show_model_training():
    st.markdown("## 🤖 Model Training & Evaluation")
    
    if 'model_results' not in st.session_state:
        with st.spinner("Training models... This may take a moment ⏳"):
            st.session_state.model_results = train_models(
                st.session_state.X_train,
                st.session_state.X_test,
                st.session_state.y_train,
                st.session_state.y_test
            )
        st.success("✅ All models trained successfully!")
    
    results = st.session_state.model_results
    
    # Model Selection
    selected_model = st.selectbox(
        "Select Model to View",
        list(results.keys())
    )
    
    st.markdown(f"### 📊 {selected_model} Performance")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{results[selected_model]['accuracy']:.4f}")
    with col2:
        st.metric("ROC-AUC Score", f"{results[selected_model]['roc_auc']:.4f}")
    with col3:
        precision = results[selected_model]['classification_report']['weighted avg']['precision']
        st.metric("Precision (avg)", f"{precision:.4f}")
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = results[selected_model]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title(f"Confusion Matrix - {selected_model}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plot_to_streamlit(fig)
    
    # Classification Report
    st.subheader("📋 Classification Report")
    
    report_df = pd.DataFrame(results[selected_model]['classification_report']).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    # ROC Curve
    st.subheader("📈 ROC Curve")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(st.session_state.y_test, results[selected_model]['y_proba'])
    ax.plot(fpr, tpr, label=f'{selected_model} (AUC = {results[selected_model]["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    plot_to_streamlit(fig)

def show_prediction():
    st.markdown("## 🔮 Make Bankruptcy Prediction")
    
    if 'model_results' not in st.session_state:
        st.warning("⚠️ Please train models first! Go to 'Model Training' page.")
        return
    
    # Model Selection
    model_name = st.selectbox(
        "Select Model",
        list(st.session_state.model_results.keys())
    )
    
    st.markdown("---")
    st.subheader("📝 Enter Risk Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5, 0.5)
        management_risk = st.slider("Management Risk", 0.0, 1.0, 0.5, 0.5)
        financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.5, 0.5)
    
    with col2:
        credibility = st.slider("Credibility", 0.0, 1.0, 0.5, 0.5)
        competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5, 0.5)
        operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5, 0.5)
    
    # Predict Button
    if st.button("🔮 Predict Bankruptcy Risk", type="primary"):
        # Prepare input
        input_data = np.array([[
            industrial_risk, management_risk, financial_flexibility,
            credibility, competitiveness, operating_risk
        ]])
        
        # Get model and scaler
        model_info = st.session_state.model_results[model_name]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Scale if necessary
        if model_info['use_scaling']:
            input_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
        
        # Display Results
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **BANKRUPTCY RISK DETECTED**")
            else:
                st.success("✅ **NO BANKRUPTCY RISK**")
        
        if proba is not None:
            with col2:
                st.metric("Bankruptcy Probability", f"{proba[1]:.2%}")
            with col3:
                st.metric("Non-Bankruptcy Probability", f"{proba[0]:.2%}")
            
            # Probability Bar Chart
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Non-Bankruptcy', 'Bankruptcy']
            colors = ['#66c2a5', '#fc8d62']
            ax.barh(categories, proba, color=colors)
            ax.set_xlabel('Probability')
            ax.set_title('Prediction Confidence')
            for i, v in enumerate(proba):
                ax.text(v + 0.02, i, f'{v:.2%}', va='center')
            plot_to_streamlit(fig)

def show_comparison():
    st.markdown("## 📉 Model Performance Comparison")
    
    if 'model_results' not in st.session_state:
        st.warning("⚠️ Please train models first! Go to 'Model Training' page.")
        return
    
    results = st.session_state.model_results
    
    # Comparison Table
    st.subheader("📊 Performance Metrics")
    
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'ROC-AUC': metrics['roc_auc'],
            'Precision': metrics['classification_report']['weighted avg']['precision'],
            'Recall': metrics['classification_report']['weighted avg']['recall'],
            'F1-Score': metrics['classification_report']['weighted avg']['f1-score']
        })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    # Accuracy Comparison Chart
    st.subheader("📈 Accuracy Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    comparison_df.set_index('Model')['Accuracy'].plot(kind='bar', color=colors, ax=ax)
    plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plot_to_streamlit(fig)
    
    # Multi-Metric Comparison
    st.subheader("🎯 Multi-Metric Comparison")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison_df))
    width = 0.15
    
    metrics = ['Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, comparison_df[metric], width, label=metric, color=colors[i])
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Across Multiple Metrics')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(comparison_df['Model'], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plot_to_streamlit(fig)
    
    # Best Model Recommendation
    st.markdown("---")
    st.subheader("🏆 Recommendation")
    
    best_model = comparison_df.iloc[0]['Model']
    best_accuracy = comparison_df.iloc[0]['Accuracy']
    
    st.success(f"""
    ### Best Performing Model: **{best_model}**
    
    - **Accuracy**: {best_accuracy:.4f}
    - **ROC-AUC**: {comparison_df.iloc[0]['ROC-AUC']:.4f}
    - **F1-Score**: {comparison_df.iloc[0]['F1-Score']:.4f}
    
    This model shows the best overall performance for bankruptcy prediction on the test set.
    """)

if __name__ == "__main__":
    main()
