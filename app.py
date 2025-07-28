import streamlit as st

# Set page config
st.set_page_config(
    page_title="BudgetFlow NLP",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 BudgetFlow NLP")
    st.markdown("### NLP-powered Union Budget Analysis Platform")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_choice = st.sidebar.selectbox(
        "Choose Analysis Tool:",
        ["🏠 Home", "📈 Trend Analyzer", "⚖️ Comparison Tool"]
    )
    
    if app_choice == "🏠 Home":
        show_home_page()
    elif app_choice == "📈 Trend Analyzer":
        show_trend_analyzer()
    elif app_choice == "⚖️ Comparison Tool":
        show_comparison_tool()

def show_home_page():
    """Display the home page with project overview"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to BudgetFlow NLP")
        st.markdown("""
        **BudgetFlow NLP** is an advanced Natural Language Processing tool for analyzing Union Budget data from 2015-2025. 
        
        ### 🎯 Key Features:
        - **📈 Trend Analysis**: Visualize budget allocation trends across years and sectors
        - **⚖️ Sector Comparison**: Compare allocations between different sectors and years
        - **🔍 Smart Data Extraction**: Automatically extracts budget amounts, years, and sectors from text
        - **📊 Interactive Visualizations**: Professional charts and graphs for insights
        
        ### 🚀 Getting Started:
        1. Select an analysis tool from the sidebar
        2. Explore different sectors and years
        3. Download data and insights
        """)
        
        st.subheader("🔧 Technologies Used")
        col_tech1, col_tech2 = st.columns(2)
        with col_tech1:
            st.markdown("""
            **NLP & AI:**
            - Sentence Transformers
            - FAISS Vector Search
            - Entity Extraction
            - Text Classification
            """)
        with col_tech2:
            st.markdown("""
            **Visualization:**
            - Streamlit
            - Matplotlib & Seaborn
            - Plotly
            - Interactive Charts
            """)
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        # Try to load some basic stats
        try:
            import pickle
            import faiss
            
            # Load basic data info
            with open("vectorstore/index.pkl", "rb") as f:
                chunks = pickle.load(f)
            
            index = faiss.read_index("vectorstore/index.faiss")
            
            st.metric("📄 Text Chunks", len(chunks))
            st.metric("🔍 Vector Embeddings", index.ntotal)
            st.metric("📅 Years Covered", "2015-2025")
            st.metric("🏛️ Data Source", "Union Budgets")
            
        except FileNotFoundError:
            st.warning("⚠️ Vector store not found. Please ensure data files are available.")
            st.info("📝 This demo requires pre-processed budget data to function.")
        
        st.subheader("🔗 Project Links")
        st.markdown("""
        - 🐙 [GitHub Repository](https://github.com/JALAJ-PATEL/BudgetFlow-NLP)
        - 📖 [Documentation](https://github.com/JALAJ-PATEL/BudgetFlow-NLP#readme)
        - ⭐ [Star on GitHub](https://github.com/JALAJ-PATEL/BudgetFlow-NLP)
        """)

def show_trend_analyzer():
    """Load the trend analyzer tool"""
    st.markdown("---")
    
    # Import and run the trend analyzer
    try:
        import sys
        import os
        
        # Add current directory to path to import the modules
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        
        # Import the trend analyzer functions
        from budget_trend_analyzer import (
            load_models, load_vectorstore, extract_budget_data, 
            normalize_amount, plot_sector_trends, compare_sectors
        )
        
        # Load data
        with st.spinner("Loading budget data..."):
            model = load_models()
            index, chunks = load_vectorstore()
            
            if index is None:
                st.error("Vector store files not found. Please ensure data files are available.")
                return
            
            df = extract_budget_data()
        
        if df is None or df.empty:
            st.error("No budget data could be extracted. Please check your data files.")
            return
        
        st.success(f"✅ Loaded {len(df)} budget records")
        
        # Display basic statistics
        st.sidebar.header("Data Overview")
        st.sidebar.metric("Total Records", len(df))
        st.sidebar.metric("Years Covered", df['year'].nunique())
        st.sidebar.metric("Sectors Identified", df['sector'].nunique())
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Trend Analysis", "📋 Data Explorer", "💡 Insights"])
        
        with tab1:
            st.header("Budget Allocation Trends")
            
            # Sector selection
            available_sectors = sorted(df['sector'].unique())
            selected_sectors = st.multiselect(
                "Select sectors to analyze:",
                available_sectors,
                default=available_sectors[:3] if len(available_sectors) >= 3 else available_sectors
            )
            
            if selected_sectors:
                fig = plot_sector_trends(df, selected_sectors)
                st.pyplot(fig)
                
                # Show trend statistics
                st.subheader("Trend Statistics")
                for sector in selected_sectors:
                    sector_data = df[df['sector'] == sector]
                    if not sector_data.empty:
                        yearly_totals = sector_data.groupby('year')['amount_normalized'].sum()
                        if len(yearly_totals) > 1:
                            growth = ((yearly_totals.iloc[-1] - yearly_totals.iloc[0]) / yearly_totals.iloc[0]) * 100
                            st.metric(
                                f"{sector} Growth", 
                                f"{growth:.1f}%",
                                f"From {yearly_totals.index[0]} to {yearly_totals.index[-1]}"
                            )
        
        with tab2:
            st.header("Raw Data Explorer")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                year_filter = st.selectbox("Filter by Year:", ["All"] + sorted(df['year'].unique()))
            with col2:
                sector_filter = st.selectbox("Filter by Sector:", ["All"] + sorted(df['sector'].unique()))
            
            # Apply filters
            filtered_df = df.copy()
            if year_filter != "All":
                filtered_df = filtered_df[filtered_df['year'] == year_filter]
            if sector_filter != "All":
                filtered_df = filtered_df[filtered_df['sector'] == sector_filter]
            
            st.dataframe(filtered_df[['year', 'sector', 'amount', 'unit', 'amount_normalized']])
            
            # Download option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"budget_data_{year_filter}_{sector_filter}.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.header("Key Insights")
            
            if not df.empty:
                # Overall statistics
                st.subheader("📈 Overall Trends")
                
                total_allocation = df.groupby('year')['amount_normalized'].sum()
                if len(total_allocation) > 1:
                    overall_growth = ((total_allocation.iloc[-1] - total_allocation.iloc[0]) / total_allocation.iloc[0]) * 100
                    st.metric("Overall Budget Growth", f"{overall_growth:.1f}%")
                
                # Sector insights
                st.subheader("🎯 Sector Insights")
                sector_growth = {}
                for sector in df['sector'].unique():
                    sector_data = df[df['sector'] == sector].groupby('year')['amount_normalized'].sum()
                    if len(sector_data) > 1:
                        growth = ((sector_data.iloc[-1] - sector_data.iloc[0]) / sector_data.iloc[0]) * 100
                        sector_growth[sector] = growth
                
                if sector_growth:
                    fastest_growing = max(sector_growth.items(), key=lambda x: x[1])
                    slowest_growing = min(sector_growth.items(), key=lambda x: x[1])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Fastest Growing Sector", fastest_growing[0], f"+{fastest_growing[1]:.1f}%")
                    with col2:
                        st.metric("Slowest Growing Sector", slowest_growing[0], f"{slowest_growing[1]:+.1f}%")
        
    except Exception as e:
        st.error(f"Error loading Trend Analyzer: {str(e)}")
        st.info("Please ensure all required files are available and try again.")

def show_comparison_tool():
    """Load the comparison tool"""
    st.markdown("---")
    
    try:
        import sys
        import os
        
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        
        # Import comparison tool functions
        from budget_comparison_tool import (
            load_models, load_vectorstore, extract_sector_allocations,
            normalize_amount, calculate_relevance, compare_sectors
        )
        
        # Load data
        with st.spinner("Loading data..."):
            model = load_models()
            index, chunks = load_vectorstore()
        
        if index is None:
            st.error("Vector store files not found. Please ensure data files are available.")
            return
        
        st.success("✅ Data loaded successfully")
        
        # Sidebar controls
        st.sidebar.header("Comparison Options")
        
        comparison_type = st.sidebar.radio(
            "Select comparison type:",
            ["Sector vs Sector", "Year vs Year", "Specific Query"]
        )
        
        if comparison_type == "Sector vs Sector":
            st.header("Compare Budget Allocations Between Sectors")
            
            available_sectors = ['Education', 'Healthcare', 'Defense', 'Agriculture', 
                               'Infrastructure', 'Digital India', 'Social Welfare', 
                               'Environment', 'Employment']
            
            col1, col2 = st.columns(2)
            with col1:
                selected_sectors = st.multiselect(
                    "Select sectors to compare:",
                    available_sectors,
                    default=['Education', 'Healthcare', 'Defense']
                )
            
            with col2:
                year_filter = st.selectbox(
                    "Filter by year (optional):",
                    ["All Years"] + [f"20{year}" for year in range(15, 26)]
                )
            
            if selected_sectors:
                year = None if year_filter == "All Years" else year_filter
                
                with st.spinner("Analyzing allocations..."):
                    comparison = compare_sectors(chunks, selected_sectors, year)
                
                if comparison:
                    st.subheader(f"Allocation Comparison {'for ' + year if year else '(All Years)'}")
                    
                    # Create comparison table
                    comparison_data = []
                    for sector, data in comparison.items():
                        comparison_data.append({
                            'Sector': sector,
                            'Amount': f"₹{data['amount']} {data['unit']}",
                            'Normalized (Crores)': f"₹{data['normalized_amount']:.2f}",
                            'Context': data['context'][:100] + "..."
                        })
                    
                    import pandas as pd
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Simple bar chart
                    if len(comparison_data) > 1:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        sectors = [item['Sector'] for item in comparison_data]
                        amounts = [comparison[sector]['normalized_amount'] for sector in sectors]
                        
                        bars = ax.bar(sectors, amounts, color='skyblue')
                        ax.set_ylabel('Amount (₹ Crores)')
                        ax.set_title(f'Budget Allocation Comparison {'- ' + year if year else ''}')
                        plt.xticks(rotation=45)
                        
                        # Add value labels
                        for bar, amount in zip(bars, amounts):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                   f'₹{amount:.0f}Cr', ha='center', va='bottom')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.warning("No allocation data found for the selected sectors and year.")
        
        elif comparison_type == "Year vs Year":
            st.header("Compare Budget Allocations Across Years")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                year1 = st.selectbox("First year:", [f"20{year}" for year in range(15, 26)])
            with col2:
                year2 = st.selectbox("Second year:", [f"20{year}" for year in range(15, 26)], index=5)
            with col3:
                sector = st.selectbox("Sector:", ['Education', 'Healthcare', 'Defense', 'Agriculture', 
                                                'Infrastructure', 'Digital India', 'Social Welfare'])
            
            if st.button("Compare Years"):
                with st.spinner("Analyzing year-wise allocations..."):
                    year1_data = extract_sector_allocations(chunks, target_year=year1, target_sector=sector)
                    year2_data = extract_sector_allocations(chunks, target_year=year2, target_sector=sector)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"{sector} in {year1}")
                    if year1_data:
                        best_y1 = year1_data[0]
                        st.metric("Allocation", f"₹{best_y1['amount']} {best_y1['unit']}")
                        st.write("**Context:**", best_y1['context'])
                    else:
                        st.write("No data found")
                
                with col2:
                    st.subheader(f"{sector} in {year2}")
                    if year2_data:
                        best_y2 = year2_data[0]
                        st.metric("Allocation", f"₹{best_y2['amount']} {best_y2['unit']}")
                        st.write("**Context:**", best_y2['context'])
                    else:
                        st.write("No data found")
                
                # Calculate change
                if year1_data and year2_data:
                    change = year2_data[0]['normalized_amount'] - year1_data[0]['normalized_amount']
                    change_pct = (change / year1_data[0]['normalized_amount']) * 100
                    
                    st.subheader("Change Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Absolute Change", f"₹{change:.2f} Crores")
                    with col2:
                        st.metric("Percentage Change", f"{change_pct:+.1f}%")
        
        else:  # Specific Query
            st.header("Specific Budget Query")
            
            query = st.text_input("Enter your specific question about budget allocations:")
            
            if query and st.button("Search"):
                with st.spinner("Searching budget documents..."):
                    # Use semantic search to find relevant chunks
                    query_vec = model.encode([query])
                    distances, indices = index.search(query_vec.astype('float32'), 5)
                    
                    results = []
                    for i, idx in enumerate(indices[0]):
                        if idx < len(chunks):
                            chunk = chunks[idx]
                            
                            # Extract amounts from this chunk
                            import re
                            amount_patterns = [
                                r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|lakh|billion)',
                                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|lakh|billion)'
                            ]
                            
                            amounts = []
                            for pattern in amount_patterns:
                                matches = re.findall(pattern, chunk, re.IGNORECASE)
                                amounts.extend(matches)
                            
                            results.append({
                                'relevance_score': 1 / (distances[0][i] + 0.1),
                                'content': chunk,
                                'amounts': amounts
                            })
                    
                    if results:
                        st.subheader("Search Results")
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i} (Relevance: {result['relevance_score']:.2f})"):
                                st.write(result['content'])
                                if result['amounts']:
                                    st.write("**Amounts mentioned:**")
                                    for amount, unit in result['amounts']:
                                        st.write(f"- ₹{amount} {unit}")
                    else:
                        st.warning("No relevant information found for your query.")
    
    except Exception as e:
        st.error(f"Error loading Comparison Tool: {str(e)}")
        st.info("Please ensure all required files are available and try again.")

if __name__ == "__main__":
    main()
