@echo off
echo ========================================
echo BudgetFlow NLP
echo ========================================
echo.
echo Available Applications:
echo 1. Budget Trend Analyzer - Visualize spending trends across years
echo 2. Budget Comparison Tool - Compare allocations between sectors/years
echo.
echo Make sure your vector store files exist in the vectorstore/ folder!
echo.

:menu
set /p choice="Enter your choice (1-2): "

if %choice%==1 (
    echo Starting Budget Trend Analyzer...
    streamlit run budget_trend_analyzer.py
    goto end
)

if %choice%==2 (
    echo Starting Budget Comparison Tool...
    streamlit run budget_comparison_tool.py
    goto end
)

echo Invalid choice. Please enter 1 or 2.
goto menu

:end
pause
