import streamlit as st
import pandas as pd
import plotly.express as px

def render_visualization_tab(df, id_to_name):
    st.title("📊 PISA Interactive Dashboard")
    st.markdown("Use the filters below to explore how economic factors influence education ratings.")

    # --- 0. PREPARATION & MAPPING ---
    plot_df = df.copy()
    if 'country' in plot_df.columns and id_to_name:
        plot_df['country_name'] = plot_df['country'].map(id_to_name)
    else:
        plot_df['country_name'] = plot_df['country'].astype(str)

    # --- 1. GLOBAL INTERACTIVE FILTERS ---
    with st.expander("🔎 Filter Data (Affects All Charts)", expanded=True):
        col_f1, col_f2 = st.columns(2)
        all_years = sorted(plot_df['time'].unique().astype(int))
        min_year, max_year = all_years[0], all_years[-1]
        
        selected_years = col_f1.slider(
            "Select Year Range:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        all_countries = sorted(plot_df['country_name'].unique())
        selected_countries = col_f2.multiselect(
            "Select Countries (Leave empty for ALL):",
            all_countries,
            default=[]
        )

    df_filtered = plot_df[
        (plot_df['time'] >= selected_years[0]) & 
        (plot_df['time'] <= selected_years[1])
    ]
    if selected_countries:
        df_filtered = df_filtered[df_filtered['country_name'].isin(selected_countries)]
    if df_filtered.empty:
        st.warning("⚠️ No data available for the selected filters.")
        return

    st.markdown("---")

    # --- 2. DYNAMIC KPI ROW ---
    avg_rating = df_filtered['rating'].mean()
    med_gdp = df_filtered['gdp_per_capita_ppp'].median()
    count_records = len(df_filtered)
    count_countries = df_filtered['country_name'].nunique()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Rating (Selection)", f"{avg_rating:.1f}")
    k2.metric("Median GDP", f"${med_gdp:,.0f}")
    k3.metric("Countries Included", count_countries)
    k4.metric("Records count", count_records)

    st.markdown("---")
    
    # --- 3. GEOSPATIAL ANALYSIS ---
    st.subheader("🌍 Geospatial Analysis")
    st.caption("Visualizing data on a world map. Darker colors indicate higher values.")
    
    geo_col1, geo_col2 = st.columns([1, 3])
    
    with geo_col1:
        map_metric = st.selectbox(
            "Select Metric to Map:",
            ['rating', 'gdp_per_capita_ppp', 'expenditure_on_education_pct_gdp','gini_index','mortality_rate_infant','population_density'],
            index=0
        )

    with geo_col2:
        map_data = df_filtered.groupby('country_name')[map_metric].mean().reset_index()
        
        fig_map = px.choropleth(
            map_data,
            locations="country_name",      # The column with ISO codes (AUS, BRA, etc.)
            color=map_metric,              # The variable to visualize
            hover_name="country_name",
            color_continuous_scale=px.colors.sequential.Plasma,
            projection="natural earth",    # 'natural earth' looks more realistic than 'equirectangular'
            title=f"Global Map of {map_metric} (Avg)"
        )
        fig_map.update_geos(showframe=False, showcoastlines=True, projection_type='equirectangular')
        fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_map, width='stretch')

    st.markdown("---")

    # --- 4. ROW 1: TIME & DISTRIBUTION ---
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("Trends Over Time")
        trend_data = df_filtered.groupby(['time', 'country_name'])['rating'].mean().reset_index()
        fig_line = px.line(
            trend_data, x='time', y='rating', color='country_name', markers=True,
            title="Rating Trajectory", labels={'rating': 'PISA Rating'}
        )
        st.plotly_chart(fig_line, width='stretch')

    with c_right:
        st.subheader("Rating Distribution")
        fig_hist = px.histogram(
            df_filtered, x="rating", nbins=20, marginal="violin",
            title="Distribution of Ratings", opacity=0.7, color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_hist, width='stretch')

    st.markdown("---")

    # --- 5. ROW 2: CORRELATIONS & SCATTER ---
    c_l, c_r = st.columns(2)

    with c_l:
        st.subheader("Economic Correlations")
        st.caption("Correlation with 'Rating' (Red = Positive, Blue = Negative)")
        cols_to_drop = ['country', 'country_name', 'time', 'rating_groups', 'sex_BOY', 'sex_GIRL', 'sex_TOT']
        corr_cols = [c for c in df_filtered.columns if c not in cols_to_drop]
        
        if len(df_filtered) > 1:
            corr_matrix = df_filtered[corr_cols].corr()
            target_corr = corr_matrix[['rating']].sort_values(by='rating', ascending=False)
            fig_heat = px.imshow(
                target_corr.T, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig_heat, width='stretch')
        else:
            st.info("Not enough data.")

    with c_r:
        st.subheader("Interactive Scatter")
        num_cols = [c for c in df_filtered.columns if c not in ['country_name', 'rating_groups']]
        x_col = st.selectbox("Select X-Axis Feature:", num_cols, index=num_cols.index('gdp_per_capita_ppp') if 'gdp_per_capita_ppp' in num_cols else 0)
        
        fig_scat = px.scatter(
            df_filtered, x=x_col, y='rating', color='country_name',
            size='population_density' if 'population_density' in df_filtered.columns else None,
            title=f"Rating vs. {x_col}", trendline="ols" if len(df_filtered) > 5 else None
        )
        st.plotly_chart(fig_scat, width='stretch')

    st.markdown("---")

    # --- 6. ROW 3: COUNTRY & GENDER COMPARISONS ---
    c_rank, c_gender = st.columns(2)

    with c_rank:
        st.subheader("Country Rankings")
        avg_by_country = df_filtered.groupby('country_name')['rating'].mean().sort_values(ascending=True).tail(15) 
        fig_rank = px.bar(
            avg_by_country, x='rating', y=avg_by_country.index, orientation='h',
            title="Top 15 Countries by Avg Rating", color='rating', color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_rank, width='stretch')

    with c_gender:
        st.subheader("Gender Performance")
        if 'sex_BOY' in df_filtered.columns:
            def get_sex(row):
                if row['sex_BOY'] == 1: return 'BOY'
                elif row['sex_GIRL'] == 1: return 'GIRL'
                elif row['sex_TOT'] == 1: return 'TOTAL'
                return 'Other'
            
            temp_gender = df_filtered.copy()
            temp_gender['Gender_Label'] = temp_gender.apply(get_sex, axis=1)
            temp_gender = temp_gender[temp_gender['Gender_Label'] != 'Other']
            
            avg_sex = temp_gender.groupby('Gender_Label')['rating'].mean().reset_index()
            
            fig_gender = px.bar(
                avg_sex, x='Gender_Label', y='rating', color='Gender_Label',
                title="Average Rating: Boys vs Girls", text_auto='.1f'
            )
            st.plotly_chart(fig_gender, width='stretch')
        else:
            st.warning("Gender data not found.")

    st.markdown("---")

    # # --- 7. ROW 4: ADVANCED 3D & VARIABILITY ---
    # c_3d, c_box = st.columns(2)

    # with c_3d:
    #     st.subheader("✨ 3D Economic View")
    #     fig_3d = px.scatter_3d(
    #         df_filtered,
    #         x='gdp_per_capita_ppp',
    #         y='expenditure_on_education_pct_gdp',
    #         z='rating',
    #         color='country_name',
    #         title="3D Analysis",
    #         opacity=0.8
    #     )
    #     st.plotly_chart(fig_3d, width='stretch')

    # with c_box:
    #     st.subheader("📦 Rating Consistency (Box Plot)")
    #     sorted_countries = df_filtered.groupby('country_name')['rating'].median().sort_values(ascending=False).index
    #     fig_box = px.box(
    #         df_filtered, x='country_name', y='rating',
    #         category_orders={'country_name': list(sorted_countries)},
    #         title="Rating Variability by Country"
    #     )
    #     st.plotly_chart(fig_box, width='stretch')