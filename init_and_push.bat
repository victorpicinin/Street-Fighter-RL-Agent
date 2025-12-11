@echo off
echo ========================================
echo Initializing Git Repository and Pushing
echo ========================================
echo.

echo Step 1: Initializing git repository...
git init
echo.

echo Step 2: Adding all files...
git add -A
echo.

echo Step 3: Checking what will be committed...
git status --short
echo.

echo Step 4: Creating initial commit...
git commit -m "Initial commit: Street Fighter RL Agent with training dashboard, database integration, and configuration management"
echo.

echo Step 5: Setting remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/victorpicinin/Street-Fighter-RL-Agent.git
echo.

echo Step 6: Setting branch to main...
git branch -M main
echo.

echo Step 7: Pushing to GitHub...
echo NOTE: You may be prompted for GitHub credentials
echo Use your GitHub username and Personal Access Token
echo.
git push -u origin main
echo.

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Code pushed to GitHub
    echo ========================================
) else (
    echo.
    echo ========================================
    echo PUSH FAILED - Check authentication
    echo ========================================
    echo You may need to:
    echo 1. Create a Personal Access Token on GitHub
    echo 2. Run: git push -u origin main
    echo 3. Enter your username and token when prompted
)

pause

