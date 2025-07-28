# BudgetFlow NLP 📊💰

An advanced NLP project for analyzing Union Budget data from 2015-2025 with specialized tools for trend analysis and sector comparisons.

## 🎯 About BudgetFlow NLP

**BudgetFlow NLP** transforms complex Union Budget documents into actionable insights using Natural Language Processing. Instead of a general-purpose chatbot, this project provides **two specialized analytical tools** that deliver reliable, measurable results for budget analysis.

### 1. Budget Trend Analyzer 📈
- **Purpose**: Visualize budget allocation trends across years
- **Features**: 
  - Multi-sector trend visualization
  - Growth statistics and percentage changes
  - Interactive charts and graphs
  - Data export functionality

### 2. Budget Comparison Tool ⚖️
- **Purpose**: Compare budget allocations between sectors and years
- **Features**:
  - Sector vs Sector comparisons
  - Year vs Year analysis
  - Percentage change calculations
  - Context-aware search

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Vector Store Exists**
   - Make sure `vectorstore/index.faiss` and `vectorstore/index.pkl` files exist
   - These contain the processed budget data embeddings

3. **Run BudgetFlow NLP**
   ```bash
   # Option 1: Unified app (RECOMMENDED FOR DEPLOYMENT)
   streamlit run app.py
   
   # Option 2: Use the launcher
   run_apps.bat
   
   # Option 3: Run individual tools
   streamlit run budget_trend_analyzer.py
   streamlit run budget_comparison_tool.py
   ```

## 📁 Project Structure

```
├── budget_trend_analyzer.py      # BudgetFlow NLP Trend Analyzer
├── budget_comparison_tool.py     # BudgetFlow NLP Comparison Tool
├── run_apps.bat                  # Application launcher
├── setup.bat                     # Easy installation script
├── requirements.txt              # Python dependencies
├── Union_Budget_2015_to_2025.txt # Source budget data
└── vectorstore/                  # Processed embeddings
    ├── index.faiss              # FAISS vector index
    └── index.pkl                # Text chunks
```

## 🔧 Technologies Used

- **NLP**: sentence-transformers, FAISS
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Interface**: Streamlit
- **Text Processing**: Regular expressions, entity extraction

## 🎯 Why This Focused Approach?

✅ **Reliable Results**: Specific tools with measurable outputs  
✅ **Easy Validation**: Visual results are easy to verify  
✅ **Professional Quality**: Business-ready visualizations  
✅ **Clear Purpose**: Each tool has a specific, well-defined function  
✅ **Better UX**: Users know exactly what to expect  

## 📊 Key NLP Features

- **Entity Extraction**: Automatically identifies budget amounts, years, and sectors
- **Text Classification**: Categorizes budget items by sector using keyword matching
- **Data Normalization**: Converts different units (lakh/crore/billion) for comparison
- **Semantic Search**: Uses embeddings to find relevant budget information
- **Pattern Recognition**: Extracts structured data from unstructured text

## 🔍 Sample Use Cases

### Trend Analyzer:
- "Show me education spending trends from 2015-2025"
- "Which sector has grown the fastest?"
- "Compare healthcare vs defense spending over time"

### Comparison Tool:
- "Compare education allocation between 2020 and 2023"
- "Which sectors got the highest allocation in 2022?"
- "Show percentage change in agriculture spending"

## 📈 Future Enhancements

- [ ] Predictive modeling for future budget trends
- [ ] Anomaly detection in budget allocations
- [ ] More sophisticated sector classification
- [ ] Export to different formats (PDF, Excel)
- [ ] Advanced filtering and search capabilities
