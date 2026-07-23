@echo off
echo ============================================
echo  Brain Tumor CNN Project - Setup Script
echo ============================================
echo.

:: Step 1: Create virtual environment
echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

:: Step 2: Install dependencies
echo [2/4] Installing dependencies (this may take a few minutes)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: Step 3: Launch Jupyter
echo [3/4] Setup complete!
echo.
echo ============================================
echo  HOW TO RUN:
echo ============================================
echo.
echo  OPTION A - VS Code:
echo    1. Open VS Code in this folder
echo    2. Open Brain_Tumor_CNN_Project.ipynb
echo    3. Select "venv" as kernel (top-right)
echo    4. Click "Run All"
echo.
echo  OPTION B - Jupyter Notebook:
echo    Run: jupyter notebook Brain_Tumor_CNN_Project.ipynb
echo.
echo ============================================
echo.
pause
