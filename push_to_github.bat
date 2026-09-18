@echo off
title Push to GitHub - 2D ResNet-Edge MTF
cd /d "%~dp0"
echo ======================================================================
echo           Pushing 2D ResNet-Edge MTF Engine to GitHub
echo ======================================================================
echo.
git push -u origin main
echo.
echo ======================================================================
pause
