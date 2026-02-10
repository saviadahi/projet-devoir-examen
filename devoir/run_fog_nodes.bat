@echo off
REM Script pour lancer les nœuds Fog avec Spark
REM Windows version

echo ========================================
echo Lancement des Noeuds Fog (Spark)
echo ========================================
echo.

REM Activer l'environnement virtuel
call venv\Scripts\activate.bat

REM Créer les répertoires nécessaires
if not exist "checkpoints" mkdir checkpoints
if not exist "models" mkdir models
if not exist "logs" mkdir logs

echo Lancement du Fog Node-1...
start "Fog Node-1" cmd /k "python fog_nodes/fog_node.py --node-id 1 --learning-rate 0.01 --update-interval 10"

timeout /t 3 /nobreak > nul

echo Lancement du Fog Node-2...
start "Fog Node-2" cmd /k "python fog_nodes/fog_node.py --node-id 2 --learning-rate 0.01 --update-interval 10"

timeout /t 3 /nobreak > nul

echo Lancement du Fog Node-3...
start "Fog Node-3" cmd /k "python fog_nodes/fog_node.py --node-id 3 --learning-rate 0.01 --update-interval 10"

echo.
echo ========================================
echo Tous les noeuds Fog sont lances!
echo ========================================
echo.
echo Appuyez sur une touche pour quitter...
pause > nul