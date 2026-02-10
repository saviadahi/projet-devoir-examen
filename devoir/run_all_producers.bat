@echo off
REM Script pour lancer plusieurs producteurs de capteurs simultanément
REM Windows version

echo ========================================
echo Lancement des producteurs de capteurs
echo ========================================
echo.

REM Activer l'environnement virtuel
call venv\Scripts\activate.bat

REM Lancer les producteurs dans des fenêtres séparées
echo Lancement du producteur Node-1...
start "Sensor Node-1" cmd /k "python producers/sensor_producer.py --node-id 1 --interval 1 --anomaly-rate 0.05"

timeout /t 2 /nobreak > nul

echo Lancement du producteur Node-2...
start "Sensor Node-2" cmd /k "python producers/sensor_producer.py --node-id 2 --interval 1 --anomaly-rate 0.08"

timeout /t 2 /nobreak > nul

echo Lancement du producteur Node-3...
start "Sensor Node-3" cmd /k "python producers/sensor_producer.py --node-id 3 --interval 1 --anomaly-rate 0.06"

echo.
echo ========================================
echo Tous les producteurs sont lances!
echo ========================================
echo.
echo Appuyez sur une touche pour quitter...
pause > nul