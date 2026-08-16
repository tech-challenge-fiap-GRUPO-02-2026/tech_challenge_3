@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   Assistente Medico Virtual - Tech Challenge Fase 3
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual nao encontrado em .venv
    echo Rode primeiro:
    echo    py -m venv .venv
    echo    .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Iniciando backend em uma nova janela...
start "Assistente Medico - Backend" cmd /k ""%~dp0.venv\Scripts\python.exe" -m src.webapp.server --no-browser"

echo Aguardando o servidor subir...
timeout /t 3 /nobreak >nul

echo Abrindo o chat no navegador...
start "" http://127.0.0.1:8765

echo.
echo Backend rodando em uma janela separada. Feche aquela janela (ou Ctrl+C nela) para encerrar.
echo Esta janela pode ser fechada.
pause
