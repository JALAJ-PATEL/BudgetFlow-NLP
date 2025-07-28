@echo off
echo ========================================
echo BudgetFlow NLP - Setup
echo ========================================
echo.
echo Installing required packages...
echo.

pip install -r requirements.txt

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo You can now run BudgetFlow NLP applications using:
echo   run_apps.bat
echo.
echo Or run them directly:
echo   streamlit run budget_trend_analyzer.py
echo   streamlit run budget_comparison_tool.py
echo.
pause
