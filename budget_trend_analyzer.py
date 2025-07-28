import os
import pandas as pd
import re
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import defaultdict
import numpy as np

# Set page config
st.set_page_config(
    page_title="BudgetFlow NLP - Trend Analyzer",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load the sentence transformer model"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_data
def load_vectorstore():
    """Load the FAISS index and chunks"""
    try:
        index = faiss.read_index("vectorstore/index.faiss")
        with open("vectorstore/index.pkl", "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    except FileNotFoundError:
        st.error("Vector store files not found. Please run the notebook first.")
        return None, None

@st.cache_data
def extract_budget_data():
    """Extract structured budget data from chunks"""
    _, chunks = load_vectorstore()
    if not chunks:
        return None
    
    budget_data = []
    
    # Patterns for extracting budget information
    amount_patterns = [
        r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|lakh|billion|thousand)',
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|lakh|billion|thousand)',
        r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|lakh|billion|thousand)'
    ]
    
    year_pattern = r'20(1[5-9]|2[0-5])'
    sector_keywords = {
        'Education': ['education', 'school', 'university', 'student', 'teacher', 'learning'],
        'Healthcare': ['health', 'medical', 'hospital', 'medicine', 'healthcare', 'ayushman'],
        'Defense': ['defense', 'defence', 'military', 'army', 'navy', 'air force'],
        'Agriculture': ['agriculture', 'farmer', 'farming', 'crop', 'rural', 'kisan'],
        'Infrastructure': ['infrastructure', 'road', 'railway', 'transport', 'highway'],
        'Digital India': ['digital', 'technology', 'IT', 'internet', 'broadband', 'cyber'],
        'Social Welfare': ['welfare', 'pension', 'social', 'women', 'child', 'sc', 'st']
    }
    
    for chunk in chunks:
        # Extract years mentioned in this chunk
        years = re.findall(year_pattern, chunk)
        years = ['20' + year for year in years]
        
        # Extract amounts
        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, chunk, re.IGNORECASE)
            amounts.extend(matches)
        
        # Determine sector
        chunk_lower = chunk.lower()
        detected_sectors = []
        for sector, keywords in sector_keywords.items():
            if any(keyword in chunk_lower for keyword in keywords):
                detected_sectors.append(sector)
        
        # Create records
        for year in years:
            for amount, unit in amounts:
                for sector in detected_sectors:
                    budget_data.append({
                        'year': year,
                        'sector': sector,
                        'amount': amount,
                        'unit': unit,
                        'amount_normalized': normalize_amount(amount, unit),
                        'context': chunk[:200]
                    })
    
    return pd.DataFrame(budget_data)

def normalize_amount(amount_str, unit):
    """Convert amounts to crores for comparison"""
    try:
        amount = float(amount_str.replace(',', ''))
        unit_lower = unit.lower()
        
        if unit_lower in ['crore', 'crores']:
            return amount
        elif unit_lower in ['lakh', 'lakhs']:
            return amount / 100
        elif unit_lower in ['billion']:
            return amount * 1000
        elif unit_lower in ['thousand']:
            return amount / 10000
        else:
            return amount
    except:
        return 0

def plot_sector_trends(df, sectors):
    """Plot trends for selected sectors"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for sector in sectors:
        sector_data = df[df['sector'] == sector]
        if not sector_data.empty:
            yearly_totals = sector_data.groupby('year')['amount_normalized'].sum()
            ax.plot(yearly_totals.index, yearly_totals.values, marker='o', label=sector, linewidth=2)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Amount (₹ Crores)')
    ax.set_title('Budget Allocation Trends by Sector')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def compare_sectors(df, year):
    """Compare sector allocations for a specific year"""
    year_data = df[df['year'] == year]
    if year_data.empty:
        return None
    
    sector_totals = year_data.groupby('sector')['amount_normalized'].sum().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sector_totals.index, sector_totals.values)
    ax.set_xlabel('Amount (₹ Crores)')
    ax.set_title(f'Budget Allocation by Sector - {year}')
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'₹{width:.0f}Cr', ha='left', va='center')
    
    plt.tight_layout()
    return fig

def main():
    st.title("📊 BudgetFlow NLP - Trend Analyzer")
    st.markdown("### Analyze budget allocation trends and compare sectors across years")
    
    # Load data
    with st.spinner("Loading budget data..."):
        model = load_models()
        index, chunks = load_vectorstore()
        
        if index is None:
            st.stop()
        
        df = extract_budget_data()
    
    if df is None or df.empty:
        st.error("No budget data could be extracted. Please check your data files.")
        st.stop()
    
    # Display basic statistics
    st.sidebar.header("Data Overview")
    st.sidebar.metric("Total Records", len(df))
    st.sidebar.metric("Years Covered", df['year'].nunique())
    st.sidebar.metric("Sectors Identified", df['sector'].nunique())
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Trend Analysis", "⚖️ Sector Comparison", "📋 Data Explorer", "💡 Insights"])
    
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
        st.header("Sector-wise Budget Comparison")
        
        # Year selection
        available_years = sorted(df['year'].unique())
        selected_year = st.selectbox("Select year for comparison:", available_years)
        
        if selected_year:
            fig = compare_sectors(df, selected_year)
            if fig:
                st.pyplot(fig)
                
                # Show top allocations
                year_data = df[df['year'] == selected_year]
                sector_totals = year_data.groupby('sector')['amount_normalized'].sum().sort_values(ascending=False)
                
                st.subheader(f"Top 5 Sectors in {selected_year}")
                for i, (sector, amount) in enumerate(sector_totals.head().items(), 1):
                    st.write(f"{i}. **{sector}**: ₹{amount:.0f} crores")
            else:
                st.warning(f"No data available for {selected_year}")
    
    with tab3:
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
    
    with tab4:
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

if __name__ == "__main__":
    main()
