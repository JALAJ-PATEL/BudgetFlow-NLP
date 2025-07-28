@echo off
echo ========================================
echo BudgetFlow NLP
echo ========================================
echo.
echo Available Applications:
echo 1. Unified App (Both tools in one) - RECOMMENDED FOR DEPLOYMENT
echo 2. Budget Trend Analyzer - Standalone
echo 3. Budget Comparison Tool - Standalone
echo.
echo Make sure your vector store files exist in the vectorstore/ folder!
echo.

:menu
set /p choice="Enter your choice (1-3): "

if %choice%==1 (
    echo Starting Unified BudgetFlow NLP App...
    streamlit run app.py
    goto end
)

if %choice%==2 (
    echo Starting Budget Trend Analyzer...
    streamlit run budget_trend_analyzer.py
    goto end
)

if %choice%==3 (
    echo Starting Budget Comparison Tool...
    streamlit run budget_comparison_tool.py
    goto end
)

echo Invalid choice. Please enter 1, 2, or 3.
goto menu

:end
pause
