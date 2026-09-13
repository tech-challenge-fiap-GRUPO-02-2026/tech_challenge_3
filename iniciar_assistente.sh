#!/usr/bin/env bash
#
# Assistente Medico Virtual - Tech Challenge Fase 3
# Sobe o backend do chat e abre a interface no navegador (Linux/Mac).
# Equivalente ao iniciar_assistente.bat (Windows).

set -euo pipefail

# Vai para o diretorio do proprio script (equivalente ao %~dp0 do .bat)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Assistente Medico Virtual - Tech Challenge Fase 3"
echo "============================================"
echo

# Descobre o python do virtualenv (venv). Aceita layout Linux/Mac e Windows.
if [ -x "venv/bin/python" ]; then
    VENV_PY="venv/bin/python"
else
    echo "[ERRO] Ambiente virtual nao encontrado em venv"
    echo "Rode primeiro:"
    echo "   python3 -m venv venv"
    echo "   venv/bin/python -m pip install -r requirements.txt"
    echo
    exit 1
fi

PORT=8765
URL="http://127.0.0.1:8765"

# Abre o navegador de forma portavel (nao falha se nenhum estiver disponivel).
open_browser() {
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL" >/dev/null 2>&1 &
    elif command -v open >/dev/null 2>&1; then
        open "$URL" >/dev/null 2>&1 &
    else
        echo "Abra manualmente no navegador: $URL"
    fi
}

# Encerra o backend ao sair (Ctrl+C ou fechar o terminal).
cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo
        echo "Encerrando o backend..."
        kill "$SERVER_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

echo "Iniciando backend na porta $PORT..."
# --no-browser: o navegador e aberto por este script, para logica identica ao .bat.
"$VENV_PY" -m src.webapp.server --port "$PORT" --no-browser &
SERVER_PID=$!

echo "Aguardando o servidor subir..."
sleep 3

echo "Abrindo o chat no navegador..."
open_browser

echo
echo "Backend rodando (PID $SERVER_PID)."
echo "Pressione Ctrl+C para encerrar."

# Mantem o script vivo enquanto o backend roda.
wait "$SERVER_PID"
