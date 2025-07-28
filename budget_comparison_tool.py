import os
import pandas as pd
import re
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
from collections import defaultdict

# Set page config
st.set_page_config(
    page_title="BudgetFlow NLP - Comparison Tool",
    page_icon="⚖️",
    layout="wide"
)

@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_data
def load_vectorstore():
    try:
        index = faiss.read_index("vectorstore/index.faiss")
        with open("vectorstore/index.pkl", "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    except FileNotFoundError:
        st.error("Vector store files not found. Please run the notebook first.")
        return None, None

def extract_sector_allocations(chunks, target_year=None, target_sector=None):
    """Extract specific sector allocations with context"""
    results = []
    
    # Enhanced patterns for better extraction
    amount_patterns = [
        r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|lakh|billion|thousand)',
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|lakh|billion|thousand)',
        r'allocation.*?₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|lakh)',
        r'budget.*?₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|lakh)'
    ]
    
    sector_keywords = {
        'Education': ['education', 'school', 'university', 'student', 'teacher', 'learning', 'sarva shiksha'],
        'Healthcare': ['health', 'medical', 'hospital', 'medicine', 'healthcare', 'ayushman', 'wellness'],
        'Defense': ['defense', 'defence', 'military', 'army', 'navy', 'air force', 'border'],
        'Agriculture': ['agriculture', 'farmer', 'farming', 'crop', 'rural', 'kisan', 'msp'],
        'Infrastructure': ['infrastructure', 'road', 'railway', 'transport', 'highway', 'metro'],
        'Digital India': ['digital', 'technology', 'IT', 'internet', 'broadband', 'cyber', 'digitization'],
        'Social Welfare': ['welfare', 'pension', 'social', 'women', 'child', 'sc', 'st', 'tribal'],
        'Environment': ['environment', 'climate', 'green', 'renewable', 'solar', 'clean'],
        'Employment': ['employment', 'skill', 'training', 'job', 'employment guarantee', 'mgnrega']
    }
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        
        # Check year filter
        if target_year:
            if target_year not in chunk:
                continue
        
        # Extract amounts
        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, chunk, re.IGNORECASE)
            amounts.extend(matches)
        
        # Determine sectors mentioned
        mentioned_sectors = []
        for sector, keywords in sector_keywords.items():
            if any(keyword in chunk_lower for keyword in keywords):
                mentioned_sectors.append(sector)
        
        # Filter by target sector
        if target_sector and target_sector not in mentioned_sectors:
            continue
        
        # Create results
        for sector in mentioned_sectors:
            for amount, unit in amounts:
                # Normalize amount
                normalized_amount = normalize_amount(amount, unit)
                
                results.append({
                    'sector': sector,
                    'amount': amount,
                    'unit': unit,
                    'normalized_amount': normalized_amount,
                    'context': chunk[:300] + "..." if len(chunk) > 300 else chunk,
                    'relevance_score': calculate_relevance(chunk_lower, sector.lower(), amount)
                })
    
    return sorted(results, key=lambda x: x['relevance_score'], reverse=True)

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

def calculate_relevance(text, sector, amount):
    """Calculate relevance score for better ranking"""
    score = 0
    
    # Higher score if sector is mentioned multiple times
    score += text.count(sector) * 2
    
    # Higher score if amount is near sector mention
    sector_pos = text.find(sector)
    amount_pos = text.find(amount)
    if sector_pos != -1 and amount_pos != -1:
        distance = abs(sector_pos - amount_pos)
        score += max(0, 100 - distance)
    
    # Higher score for specific allocation keywords
    allocation_keywords = ['allocation', 'allocated', 'budget', 'provision', 'outlay']
    for keyword in allocation_keywords:
        if keyword in text:
            score += 10
    
    return score

def compare_sectors(chunks, sectors, year=None):
    """Compare allocations between multiple sectors"""
    comparison_results = {}
    
    for sector in sectors:
        allocations = extract_sector_allocations(chunks, target_year=year, target_sector=sector)
        if allocations:
            # Get the highest allocation for this sector
            best_allocation = max(allocations, key=lambda x: x['normalized_amount'])
            comparison_results[sector] = best_allocation
    
    return comparison_results

def main():
    st.title("⚖️ BudgetFlow NLP - Comparison Tool")
    st.markdown("### Compare budget allocations between sectors and years")
    
    # Load data
    with st.spinner("Loading data..."):
        model = load_models()
        index, chunks = load_vectorstore()
    
    if index is None:
        st.stop()
    
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
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualization
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

if __name__ == "__main__":
    main()
