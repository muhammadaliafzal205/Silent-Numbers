import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

@st.cache_data(max_entries=10, ttl=3600)
def load_data(file):
    try:
        df = pd.read_csv(file, low_memory=True)
        if df.shape[0] > 100000:
            st.warning(f"Dataset is large ({df.shape[0]} rows). Using first 100,000 rows for analysis.")
            df = df.iloc[:100000].copy()
        return df, None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"
@st.cache_data(max_entries=10, ttl=3600)
def get_correlation_matrix(df, max_cols=30):
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
    if len(numeric_cols) > 1:
        return df[numeric_cols].corr()
    return None
@st.cache_data(max_entries=10, ttl=3600)
def get_missing_values(df):
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        return missing_values[missing_values > 0]
    return None
def train_regression_model(_X_train, _y_train, model_type, alpha=1.0, l1_ratio=0.5, random_state=42):
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Ridge Regression":
        model = Ridge(alpha=alpha, random_state=random_state)
    elif model_type == "Lasso Regression":
        model = Lasso(alpha=alpha, random_state=random_state)
    elif model_type == "Elastic Net":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    model.fit(_X_train, _y_train)
    return model
def calculate_vif(X, max_cols=30):
    try:
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.shape[1] > max_cols:
            X_numeric = X_numeric.iloc[:, :max_cols]
        if X_numeric.shape[1] < 2:
            return pd.DataFrame({"Variable": X_numeric.columns, "VIF": [1.0]})
        variances = X_numeric.var()
        valid_cols = X_numeric.columns[variances > 1e-10]
        if len(valid_cols) < 2:
            return pd.DataFrame({"Variable": X_numeric.columns, 
                                "VIF": ["Low variance - VIF not applicable" 
                                       if var <= 1e-10 else 1.0 
                                       for col, var in zip(X_numeric.columns, variances)]})
        X_for_vif = X_numeric[valid_cols]
        vif_data = pd.DataFrame()
        vif_data["Variable"] = valid_cols
        vif_data["VIF"] = [variance_inflation_factor(X_for_vif.values, i) 
                           for i in range(X_for_vif.shape[1])]
        return vif_data.sort_values("VIF", ascending=False)
    except Exception as e:
        st.warning(f"VIF calculation error: {str(e)}")
        return None
class DataPreprocessor:
    def __init__(self, scale_features=False, handle_missing="drop"):
        self.scale_features = scale_features
        self.handle_missing = handle_missing
        self.numeric_transformer = None
        self.categorical_transformer = None
        self.preprocessor = None
        self.cat_features = None
        self.num_features = None
        self.feature_names = None
        self.original_data_columns = None
        self.encoder_categories_ = None  
        self.encoded_feature_names = []  
    def fit_transform(self, X, y):
        self.original_data_columns = X.columns.tolist()
        self.cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        impute_strategy = 'mean' if self.handle_missing == 'mean' else 'median'
        if self.scale_features:
            self.numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=impute_strategy)),
                ('scaler', StandardScaler())
            ])
        else:
            self.numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=impute_strategy))
            ])
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'))
        ])
        transformers = []
        if self.num_features:
            transformers.append(('num', self.numeric_transformer, self.num_features))
        if self.cat_features:
            transformers.append(('cat', self.categorical_transformer, self.cat_features))
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        X_processed = self.preprocessor.fit_transform(X)
        if self.cat_features:
            self._store_encoder_categories()
        self._generate_feature_names()
        return pd.DataFrame(X_processed, columns=self.encoded_feature_names), y
    def transform(self, X):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        missing_cols = set(self.original_data_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = np.nan
        X = X[self.original_data_columns]
        X_processed = self.preprocessor.transform(X)
        return pd.DataFrame(X_processed, columns=self.encoded_feature_names)
    def _store_encoder_categories(self):
        self.encoder_categories_ = []
        if self.cat_features:
            for name, transformer, features in self.preprocessor.transformers_:
                if name == 'cat':
                    self.encoder_categories_ = transformer.named_steps['onehot'].categories_
                    break
    def _generate_feature_names(self):
        self.encoded_feature_names = []
        if self.num_features:
            self.encoded_feature_names.extend(self.num_features)
        if self.cat_features and hasattr(self, 'encoder_categories_'):
            for i, feature in enumerate(self.cat_features):
                categories = self.encoder_categories_[i]
                if self.categorical_transformer.named_steps['onehot'].drop == 'first':
                    categories = categories[1:]
                for category in categories:
                    self.encoded_feature_names.append(f"{feature}_{category}")
def prepare_data(df, features, target, scale_features=False, handle_missing="drop"):
    if not features:
        st.error("No features selected.")
        return None, None, None, None
    if target not in df.columns:
        st.error(f"Target column '{target}' not found in dataset.")
        return None, None, None, None
    for feature in features:
        if feature not in df.columns:
            st.error(f"Feature column '{feature}' not found in dataset.")
            return None, None, None, None
    X = df[features].copy()
    y = df[target].copy()
    missing_target_mask = y.isna()
    if missing_target_mask.any():
        X = X[~missing_target_mask]
        y = y[~missing_target_mask]
        st.info(f"Dropped {missing_target_mask.sum()} rows with missing target values")
    if handle_missing == "drop":
        X_missing_mask = X.isna().any(axis=1)
        if X_missing_mask.any():
            X = X[~X_missing_mask]
            y = y[~X_missing_mask]
            st.info(f"Dropped {X_missing_mask.sum()} rows with missing feature values")
    preprocessor = DataPreprocessor(scale_features, handle_missing)
    X_processed, y = preprocessor.fit_transform(X, y)
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    return X_processed, y, preprocessor, cat_features
if 'regression_results_ready' not in st.session_state:
    st.session_state.regression_results_ready = False
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
def run():
    st.title("📊 Advanced Regression Model Web App")
    st.markdown("Upload your dataset, select variables, and get comprehensive regression analysis results.")
    st.info("💡 For large datasets (>100K rows), the app will use only the first 100K rows to ensure performance.")
    with st.sidebar:
        st.header("Model Configuration")
        regression_type = st.selectbox(
            "Select Regression Model",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"]
        )
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random State", 0, 100, 42)
        alpha = 1.0
        l1_ratio = 0.5
        if regression_type != "Linear Regression":
            alpha = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0, 0.01)
            if regression_type == "Elastic Net":
                l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01)
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="file_uploader")
    if uploaded_file is not None:
        df, error = load_data(uploaded_file)
        if error:
            st.error(error)
            return 
        st.write("### Data Overview")
        col1, col2 = st.columns(2) 
        with col1:
            st.write(f"*Rows: {df.shape[0]}, **Columns*: {df.shape[1]}")
            st.write("*Preview:*")
            st.dataframe(df.head(5), use_container_width=True) 
        with col2:
            missing = get_missing_values(df)
            if missing is not None:
                st.warning("Dataset contains missing values")
                st.write("Missing value counts:")
                st.write(missing) 
                handle_missing_option = st.selectbox("Handle missing values:", 
                                                ["Drop rows", "Fill with mean", "Fill with median"]) 
                if handle_missing_option == "Drop rows":
                    handle_missing = "drop"
                elif handle_missing_option == "Fill with mean":
                    handle_missing = "mean"
                else:
                    handle_missing = "median"
            else:
                handle_missing = "drop"
        with st.expander("Data Analysis", expanded=False):
            summary_tab, corr_tab = st.tabs(["Summary Statistics", "Correlation Matrix"]) 
            with summary_tab:
                st.write(df.describe()) 
            with corr_tab:
                corr = get_correlation_matrix(df, max_cols=30)
                if corr is not None:
                    try:
                        fig = px.imshow(corr,
                                    labels=dict(color="Correlation"),
                                    x=corr.columns,
                                    y=corr.columns,
                                    color_continuous_scale="RdBu_r") 
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating correlation heatmap: {str(e)}")
                        st.write("Displaying correlation values instead:")
                        st.dataframe(corr)
                else:
                    st.info("Not enough numerical columns for correlation analysis") 
        st.write("### Model Configuration")
        target_options = df.select_dtypes(include=[np.number]).columns
        if len(target_options) == 0:
            st.error("No numerical columns found for regression target.")
            return 
        target = st.selectbox("Select target variable (what you want to predict):", target_options) 
        features = st.multiselect(
            "Select input variables (features):",
            [col for col in df.columns if col != target]
        ) 
        scale_features = st.checkbox("Standardize features (recommended for regularized models)") 
        limit_features = st.checkbox("Limit number of features (recommended for large datasets)", 
                                     value=len(features) > 30) 
        if len(features) > 0:
            if len(features) > 30 and not limit_features:
                st.warning(f"You've selected {len(features)} features. This may cause memory issues. Consider limiting features or enabling the feature limit option.") 
            run_button = st.button("Run Regression Analysis") 
            if run_button or st.session_state.regression_results_ready:
                st.session_state.regression_results_ready = True 
                try:
                    progress_bar = st.progress(0) 
                    selected_features = features
                    if limit_features and len(features) > 30:
                        st.info(f"Limiting analysis to first 30 features to prevent memory issues")
                        selected_features = features[:30] 
                    progress_bar.progress(10, text="Preparing data...")
                    X, y, preprocessor, cat_features = prepare_data(df, selected_features, target, scale_features, handle_missing) 
                    if X is None or y is None:
                        return 
                    st.session_state.preprocessor = preprocessor
                    st.session_state.feature_names = X.columns.tolist() 
                    progress_bar.progress(20, text="Processing features...")
                    if len(cat_features) > 0:
                        st.info(f"Encoded categorical features: {', '.join(cat_features)}") 
                    performance_tab, features_tab, predictions_tab, diagnostics_tab = st.tabs([
                        "Model Performance", "Feature Analysis", "Predictions", "Diagnostics"
                    ]) 
                    progress_bar.progress(30, text="Splitting data...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    ) 
                    progress_bar.progress(40, text="Training model...")
                    if regression_type == "Linear Regression":
                        model = train_regression_model(X_train, y_train, regression_type, random_state=random_state)
                    elif regression_type == "Elastic Net":
                        model = train_regression_model(X_train, y_train, regression_type, alpha, l1_ratio, random_state)
                    else:
                        model = train_regression_model(X_train, y_train, regression_type, alpha, random_state=random_state) 
                    st.session_state.model = model 
                    progress_bar.progress(50, text="Making predictions...")
                    y_pred = model.predict(X_test) 
                    r2 = r2_score(y_test, y_pred)
                    adjusted_r2 = 1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1))
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred) 
                    progress_bar.progress(60, text="Running cross-validation...")
                    try:
                        n_folds = min(5, len(X))
                        if len(X) > 10000:
                            n_folds = min(3, n_folds) 
                        cv_scores = cross_val_score(model, X, y, cv=n_folds, scoring='r2')
                    except MemoryError:
                        st.warning("Memory limit reached during cross-validation. Skipping this step.")
                        cv_scores = np.array([r2])
                    with performance_tab:
                        st.subheader("📈 Model Performance") 
                        metric_cols = st.columns(3)
                        with metric_cols[0]:
                            st.metric("R² Score", f"{r2:.4f}")
                            st.metric("CV R² (avg)", f"{cv_scores.mean():.4f}")
                        with metric_cols[1]:
                            st.metric("Adjusted R²", f"{adjusted_r2:.4f}")
                            st.metric("Mean Absolute Error", f"{mae:.4f}")
                        with metric_cols[2]:
                            st.metric("Mean Squared Error", f"{mse:.4f}")
                            st.metric("Root MSE", f"{rmse:.4f}") 
                        result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                        if len(result_df) > 1000:
                            result_df = result_df.sample(1000, random_state=random_state) 
                        fig = px.scatter(result_df, x="Actual", y="Predicted", 
                                     title="Actual vs Predicted Values") 
                        min_val = min(result_df["Actual"].min(), result_df["Predicted"].min())
                        max_val = max(result_df["Actual"].max(), result_df["Predicted"].max())
                        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                             mode="lines", name="Perfect Prediction",
                                             line=dict(color="red", dash="dash"))) 
                        st.plotly_chart(fig, use_container_width=True)
                        residuals = y_test - y_pred
                        res_df = pd.DataFrame({"Predicted": y_pred, "Residuals": residuals}) 
                        if len(res_df) > 1000:
                            res_df = res_df.sample(1000, random_state=random_state) 
                        res_cols = st.columns(2) 
                        with res_cols[0]:
                            fig = px.scatter(res_df, x="Predicted", y="Residuals",
                                         title="Residuals vs Predicted")
                            fig.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig, use_container_width=True) 
                        with res_cols[1]:
                            fig = px.histogram(res_df, x="Residuals", 
                                           title="Residuals Distribution")
                            st.plotly_chart(fig, use_container_width=True) 
                    progress_bar.progress(80, text="Analyzing features...")
                    with features_tab:
                        st.subheader("🔍 Feature Analysis") 
                        try:
                            coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
                            coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
                            coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)
                            if len(coef_df) > 30:
                                st.info(f"Showing top 30 of {len(coef_df)} features by coefficient magnitude")
                                display_coef_df = coef_df.head(30)
                            else:
                                display_coef_df = coef_df 
                            fig = px.bar(display_coef_df, x="Feature", y="Coefficient", 
                                      title="Feature Coefficients",
                                      color="Coefficient", 
                                      color_continuous_scale=px.colors.diverging.RdBu_r) 
                            if len(display_coef_df) > 10 or display_coef_df["Feature"].str.len().max() > 15:
                                fig.update_layout(
                                    xaxis_tickangle=-45,
                                    margin=dict(l=20, r=20, t=60, b=140)
                                ) 
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying coefficients: {str(e)}") 
                        corr_cols = st.columns(2) 
                        with corr_cols[0]:
                            st.subheader("Correlation with Target")
                            try:
                                corr_with_target = pd.DataFrame()
                                for feature in selected_features[:30]:
                                    if pd.api.types.is_numeric_dtype(df[feature]):
                                        corr_with_target.loc[feature, "Correlation"] = df[feature].corr(df[target])
                                if not corr_with_target.empty:
                                    corr_with_target = corr_with_target.sort_values("Correlation", ascending=False)
                                    st.dataframe(corr_with_target)
                                else:
                                    st.info("No numeric features to calculate correlations")
                            except Exception as e:
                                st.error(f"Error calculating correlations: {str(e)}") 
                        with corr_cols[1]:
                            st.subheader("Multicollinearity (VIF)")
                            try:
                                vif_df = calculate_vif(X, max_cols=30)
                                if vif_df is not None:
                                    def highlight_vif(val):
                                        try:
                                            float_val = float(val)
                                            if float_val > 10:
                                                return 'background-color: #FFB3B3'
                                            elif float_val > 5:
                                                return 'background-color: #FFFFB3'
                                            return ''
                                        except:
                                            return '' 
                                    st.write("VIF > 5: potential multicollinearity")
                                    st.write("VIF > 10: severe multicollinearity")
                                    st.dataframe(vif_df.style.applymap(highlight_vif, subset=['VIF']))
                                else:
                                    st.info("Could not calculate VIF - requires multiple numeric features")
                            except Exception as e:
                                st.warning(f"VIF calculation error: {str(e)}")
                    progress_bar.progress(90, text="Setting up prediction interface...")
                    with predictions_tab:
                        st.subheader("🔮 Make Predictions") 
                        with st.form(key='prediction_form'):
                            prediction_inputs = {} 
                            num_cols = 2
                            if len(selected_features) > 10:
                                num_cols = 3 
                            pred_cols = st.columns(num_cols)
                            col_idx = 0 
                            for feature in selected_features:
                                with pred_cols[col_idx]:
                                    if pd.api.types.is_numeric_dtype(df[feature]):
                                        prediction_inputs[feature] = st.number_input(
                                            f"{feature}", 
                                            value=float(df[feature].median()),
                                            format="%.4f",
                                            key=f"pred_{feature}"
                                        )
                                    else:
                                        unique_values = df[feature].dropna().unique().tolist() 
                                        if len(unique_values) > 100:
                                            unique_values = unique_values[:100]
                                            st.warning(f"Showing only first 100 values for {feature}") 
                                        prediction_inputs[feature] = st.selectbox(
                                            f"{feature}",
                                            options=unique_values,
                                            key=f"pred_{feature}"
                                        ) 
                                col_idx = (col_idx + 1) % num_cols 
                            predict_button = st.form_submit_button("Make Prediction") 
                        if predict_button:
                            try:
                                input_df = pd.DataFrame([prediction_inputs]) 
                                if st.session_state.preprocessor is not None and st.session_state.model is not None:
                                    X_pred = st.session_state.preprocessor.transform(input_df) 
                                    prediction = st.session_state.model.predict(X_pred)[0] 
                                    st.success(f"Predicted {target}: *{prediction:.4f}*") 
                                    if st.session_state.feature_names and hasattr(st.session_state.model, 'coef_'):
                                        st.write("### Feature Contributions")
                                        contributions = pd.DataFrame({
                                            'Feature': st.session_state.feature_names,
                                            'Value': X_pred.iloc[0].values,
                                            'Coefficient': st.session_state.model.coef_,
                                            'Contribution': X_pred.iloc[0].values * st.session_state.model.coef_
                                        }) 
                                        if hasattr(st.session_state.model, 'intercept_'):
                                            intercept_row = pd.DataFrame({
                                                'Feature': ['Intercept'],
                                                'Value': [1],
                                                'Coefficient': [st.session_state.model.intercept_],
                                                'Contribution': [st.session_state.model.intercept_]
                                            })
                                            contributions = pd.concat([intercept_row, contributions]) 
                                        contributions['Abs_Contribution'] = contributions['Contribution'].abs()
                                        contributions = contributions.sort_values('Abs_Contribution', ascending=False) 
                                        st.dataframe(contributions[['Feature', 'Value', 'Coefficient', 'Contribution']]) 
                                        total_contribution = contributions['Contribution'].sum()
                                        st.metric("Sum of contributions", f"{total_contribution:.4f}")
                                        st.metric("Prediction", f"{prediction:.4f}")
                                else:
                                    st.error("Model or preprocessor not found. Please run the regression analysis first.")
                            except Exception as e:
                                st.error(f"Prediction error: {str(e)}")
                                st.error("Make sure all features have valid values") 
                    progress_bar.progress(95, text="Generating diagnostic plots...")
                    with diagnostics_tab:
                        st.subheader("🔍 Model Diagnostics") 
                        if len(X) <= 10000:
                            try:
                                X_sm = sm.add_constant(X) 
                                model_sm = sm.OLS(y, X_sm).fit() 
                                with st.expander("Detailed Statistics Summary"):
                                    st.text(model_sm.summary()) 
                                residuals_sm = model_sm.resid
                                fig = go.Figure() 
                                residuals_sorted = np.sort(residuals_sm)
                                theoretical_quantiles = stats.norm.ppf(
                                    np.linspace(0.01, 0.99, len(residuals_sorted))
                                )
                                fig.add_trace(go.Scatter(
                                    x=theoretical_quantiles,
                                    y=residuals_sorted,
                                    mode='markers',
                                    marker=dict(size=6),
                                    name='Residuals'
                                ))
                                min_val = min(theoretical_quantiles)
                                max_val = max(theoretical_quantiles)
                                fig.add_trace(go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val * np.std(residuals_sm), max_val * np.std(residuals_sm)],
                                    mode='lines',
                                    line=dict(color='red', dash='dash'),
                                    name='Reference Line'
                                )) 
                                fig.update_layout(
                                    title='Normal Q-Q Plot',
                                    xaxis_title='Theoretical Quantiles',
                                    yaxis_title='Sample Quantiles'
                                ) 
                                st.plotly_chart(fig, use_container_width=True) 
                                standardized_residuals = residuals_sm / np.sqrt(np.var(residuals_sm))
                                abs_sqrt_std_resid = np.sqrt(np.abs(standardized_residuals)) 
                                fig = px.scatter(
                                    x=model_sm.fittedvalues,
                                    y=abs_sqrt_std_resid,
                                    title="Scale-Location Plot",
                                    labels={
                                        'x': 'Fitted values',
                                        'y': '√|Standardized residuals|'
                                    }
                                ) 
                                try:
                                    if len(model_sm.fittedvalues) > 1000:
                                        idx = np.random.choice(len(model_sm.fittedvalues), 1000, replace=False)
                                        x_lowess = np.array(model_sm.fittedvalues)[idx]
                                        y_lowess = np.array(abs_sqrt_std_resid)[idx]
                                    else:
                                        x_lowess = model_sm.fittedvalues
                                        y_lowess = abs_sqrt_std_resid 
                                    lowess = sm.nonparametric.lowess(y_lowess, x_lowess, frac=0.5) 
                                    lowess_sorted = lowess[lowess[:, 0].argsort()] 
                                    fig.add_trace(go.Scatter(
                                        x=lowess_sorted[:, 0],
                                        y=lowess_sorted[:, 1],
                                        mode='lines',
                                        line=dict(color='red', width=2),
                                        name='LOWESS'
                                    ))
                                except Exception as e:
                                    st.warning(f"Could not generate LOWESS line: {str(e)}") 
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error in diagnostic plots: {str(e)}")
                                st.info("Detailed diagnostics are not available for this model")
                        else:
                            st.info("Detailed diagnostics are only available for datasets with 10,000 or fewer rows") 
                        st.write("### Model Validation")
                        st.write(f"*Cross-validation R² scores*: {', '.join([f'{score:.4f}' for score in cv_scores])}")
                        st.write(f"*Cross-validation R² mean*: {cv_scores.mean():.4f}")
                        st.write(f"*Cross-validation R² std*: {cv_scores.std():.4f}") 
                    progress_bar.progress(100, text="Analysis complete!") 
                except MemoryError:
                    st.error("Memory limit reached. Try reducing the number of features or rows in your dataset.")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
