import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson
import numpy as np
# Set page configuration

def run():
    # Title
    st.title("Global Suicide Data Analysis (1985–2010)")
    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset CSV", type="csv")
    if uploaded_file is not None:
        # Read and clean data
        df = pd.read_csv(uploaded_file)
        # Clean GDP column
        if 'gdp_for_year ($) ' in df.columns:
            df['gdp_for_year ($) '] = df['gdp_for_year ($) '].astype(str).str.replace(',', '')
            df['gdp_for_year ($) '] = pd.to_numeric(df['gdp_for_year ($) '], errors='coerce')
        # Ensure categorical columns are strings
        for col in ['sex', 'age', 'generation', 'country']:
            df[col] = df[col].astype(str)
    
        # Clean HDI column
        df['HDI for year'] = pd.to_numeric(df['HDI for year'], errors='coerce')
        # Sidebar filters
        st.sidebar.header("Filters")
        countries = sorted(df['country'].unique())
        selected_countries = st.sidebar.multiselect("Select Countries", countries, default=["Albania"])
        year_min, year_max = st.sidebar.slider("Select Year Range", int(df['year'].min()), int(df['year'].max()), (1985, 2010))
        sex_options = sorted(df['sex'].unique())
        selected_sex = st.sidebar.multiselect("Select Gender", sex_options, default=sex_options)
        age_groups = sorted(df['age'].unique())
        selected_age = st.sidebar.multiselect("Select Age Group", age_groups, default=age_groups)
        generations = sorted(df['generation'].unique())
        selected_gen = st.sidebar.multiselect("Select Generation", generations, default=generations)
        # Filter data
        filtered_df = df[
            (df['country'].isin(selected_countries)) &
            (df['year'].between(year_min, year_max)) &
            (df['sex'].isin(selected_sex)) &
            (df['age'].isin(selected_age)) &
            (df['generation'].isin(selected_gen))
        ]
        if filtered_df.empty:
            st.warning("No data matches the selected filters.")
        else:
            # Tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Demographic Analysis",
                "Happiest Countries",
                "Temporal Trends",
                "Economic Correlations",
                "Poisson Distribution"
            ])
    
            with tab1:
                st.header("Demographic Analysis")
                # Suicide rates by sex and age
                demo_df = filtered_df.groupby(['sex', 'age'])['suicides/100k pop'].mean().reset_index()
                fig1 = px.bar(
                    demo_df,
                    x='age',
                    y='suicides/100k pop',
                    color='sex',
                    barmode='group',
                    title="Average Suicide Rates by Sex and Age Group",
                    labels={'suicides/100k pop': 'Suicide Rate (per 100,000)'}
                )
                st.plotly_chart(fig1, use_container_width=True)
                # Suicide rates by generation
                gen_df = filtered_df.groupby('generation')['suicides/100k pop'].describe().reset_index()
                fig2 = px.box(
                    filtered_df,
                    x='generation',
                    y='suicides/100k pop',
                    title="Suicide Rate Distribution by Generation",
                    labels={'suicides/100k pop': 'Suicide Rate (per 100,000)'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                # Insights
                st.write("""
                *Insights*:
                - Males typically have higher suicide rates than females across age groups.
                - Older age groups (e.g., 75+) and younger groups (e.g., 15-24) often show elevated rates.
                - Generations like Generation X and Boomers may have higher median rates due to socioeconomic factors.
                """)
    
            with tab2:
                st.header("Happiest Countries (Lowest Suicide Rates)")
                # Average suicide rate per country
                country_rates = df.groupby('country')['suicides/100k pop'].mean().reset_index()
                happiest_countries = country_rates.sort_values('suicides/100k pop').head(10)
                fig3 = px.bar(
                    happiest_countries,
                    x='country',
                    y='suicides/100k pop',
                    title="Top 10 Countries with Lowest Suicide Rates",
                    labels={'suicides/100k pop': 'Average Suicide Rate (per 100,000)'},
                    color='suicides/100k pop',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig3, use_container_width=True)
                # Download happiest countries data
                happiest_csv = happiest_countries.to_csv(index=False)
                st.download_button(
                    label="Download Happiest Countries Data",
                    data=happiest_csv,
                    file_name="happiest_countries.csv",
                    mime="text/csv"
                )
                # Interesting fact
                happiest_country = happiest_countries.iloc[0]['country']
                lowest_rate = happiest_countries.iloc[0]['suicides/100k pop']
                st.write(f"*Interesting Fact*: {happiest_country} has the lowest average suicide rate ({lowest_rate:.2f} per 100,000), suggesting high well-being or underreporting.")
            with tab3:
                st.header("Temporal Trends in Suicide Rates")
                # Select country for trend
                trend_country = st.selectbox("Select Country for Trend", countries, index=countries.index("Albania"))
                trend_df = df[df['country'] == trend_country].groupby('year')['suicides/100k pop'].mean().reset_index()
                fig4 = px.line(
                    trend_df,
                    x='year',
                    y='suicides/100k pop',
                    title=f"Suicide Rate Trend in {trend_country}",
                    labels={'suicides/100k pop': 'Suicide Rate (per 100,000)', 'year': 'Year'},
                    markers=True
                )
                st.plotly_chart(fig4, use_container_width=True)
                # Calculate trend
                if len(trend_df) > 1:
                    rate_change = trend_df['suicides/100k pop'].iloc[-1] - trend_df['suicides/100k pop'].iloc[0]
                    st.write(f"*Trend Insight*: Suicide rate in {trend_country} {'increased' if rate_change > 0 else 'decreased'} by {abs(rate_change):.2f} per 100,000 from {trend_df['year'].iloc[0]} to {trend_df['year'].iloc[-1]}.")
                else:
                    st.write("Insufficient data for trend analysis.")
            with tab4:
                st.header("Economic and Development Correlations")
                # Aggregate data by country and year
                econ_df = df.groupby(['country', 'year']).agg({
                    'suicides/100k pop': 'mean',
                    'gdp_per_capita ($)': 'mean',
                    'HDI for year': 'mean'
                }).reset_index()
                # Scatter: Suicide rate vs GDP per capita
                fig5 = px.scatter(
                    econ_df,
                    x='gdp_per_capita ($)',
                    y='suicides/100k pop',
                    color='country',
                    hover_data=['year'],
                    title="Suicide Rate vs. GDP Per Capita",
                    labels={'suicides/100k pop': 'Suicide Rate (per 100,000)', 'gdp_per_capita ($)': 'GDP Per Capita ($)'}
                )
                st.plotly_chart(fig5, use_container_width=True)
                # Correlation
                corr_gdp = econ_df['suicides/100k pop'].corr(econ_df['gdp_per_capita ($)'])
                st.write(f"*Correlation with GDP Per Capita*: {corr_gdp:.2f}")
                # Scatter: Suicide rate vs HDI
                hdi_df = econ_df.dropna(subset=['HDI for year'])
                if not hdi_df.empty:
                    fig6 = px.scatter(
                        hdi_df,
                        x='HDI for year',
                        y='suicides/100k pop',
                        color='country',
                        hover_data=['year'],
                        title="Suicide Rate vs. HDI",
                        labels={'(shader://colorscales/viridis': 'Suicide Rate (per 100,000)', 'HDI for year': 'HDI'}
                    )
                    st.plotly_chart(fig6, use_container_width=True)
                    corr_hdi = hdi_df['suicides/100k pop'].corr(hdi_df['HDI for year'])
                    st.write(f"*Correlation with HDI*: {corr_hdi:.2f}")
                else:
                    st.write("No HDI data available for correlation.")
                st.write("""
                *Insights*:
                - Negative correlation with GDP per capita suggests wealthier countries may have lower suicide rates.
                - HDI correlation varies; high HDI doesn't always imply low suicide rates.
                """)
            with tab5:
                st.header("Poisson Distribution Analysis")
                # Select specific demographic group
                poisson_sex = st.selectbox("Select Gender for Poisson", sex_options)
                poisson_age = st.selectbox("Select Age Group for Poisson", age_groups)
                poisson_df = filtered_df[(filtered_df['sex'] == poisson_sex) & (filtered_df['age'] == poisson_age)]
                if poisson_df['suicides_no'].empty:
                    st.warning("No data for selected demographic group.")
                else:
                    lambda_ = poisson_df['suicides_no'].mean()
                    st.write(f"*Mean number of suicides (λ)*: {lambda_:.2f}")
                    # Poisson PMF
                    x_vals = list(range(0, int(poisson_df['suicides_no'].max()) + 5))
                    pmf_vals = poisson.pmf(x_vals, mu=lambda_)
                    fig7 = go.Figure()
                    fig7.add_trace(go.Bar(x=x_vals, y=pmf_vals, name="Poisson PMF", marker_color='skyblue'))
                    fig7.update_layout(
                        title="Poisson Distribution PMF",
                        xaxis_title="Suicide Count",
                        yaxis_title="Probability"
                    )
                    st.plotly_chart(fig7, use_container_width=True)
                    # Histogram vs Poisson
                    hist_data = poisson_df['suicides_no']
                    bins = range(0, int(hist_data.max()) + 2)
                    hist_vals, bin_edges = np.histogram(hist_data, bins=bins, density=True)
                    fig8 = go.Figure()
                    fig8.add_trace(go.Bar(x=bin_edges[:-1], y=hist_vals, name="Actual", opacity=0.6))
                    fig8.add_trace(go.Scatter(x=x_vals, y=pmf_vals, mode='lines+markers', name="Poisson Fit", line=dict(color='red', dash='dash')))
                    fig8.update_layout(
                        title="Actual vs Poisson Distribution",
                        xaxis_title="Suicide Count",
                        yaxis_title="Density",
                        showlegend=True
                    )
                    st.plotly_chart(fig8, use_container_width=True)
    
                    # Probability calculation
                    value = st.slider("Select X to calculate P(X = x)", 0, int(poisson_df['suicides_no'].max()), key="poisson_slider")
                    prob = poisson.pmf(value, mu=lambda_)
                    st.write(f"*Probability of exactly {value} suicides (P(X={value}))*: {prob:.4f}")
    else:
        st.info("Please upload a CSV file to begin analysis.")
