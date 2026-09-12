#!/usr/bin/env bash
#
# Instala o PyTorch com suporte a CUDA (GPUs NVIDIA) + dependências de
# fine-tuning, dentro do virtualenv do projeto.
#
# Uso:
#   ./scripts/install_torch_nvidia.sh
#
# Variáveis de ambiente opcionais:
#   CUDA_CHANNEL   Canal de wheels do PyTorch (padrão: cu121).
#                  Ex.: cu118, cu121, cu124 — veja https://pytorch.org
#   VENV_DIR       Caminho do virtualenv (padrão: venv na raiz do projeto).
#
set -euo pipefail

# Raiz do projeto = diretório pai da pasta deste script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_CHANNEL="${CUDA_CHANNEL:-cu121}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/venv}"

echo "==> Projeto:        ${PROJECT_ROOT}"
echo "==> Virtualenv:     ${VENV_DIR}"
echo "==> Canal CUDA:     ${CUDA_CHANNEL}"

# --- virtualenv -------------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "==> Virtualenv não encontrado. Criando em ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# --- checagem de versão do Python ------------------------------------------
# O PyTorch publica wheels até Python 3.13. Em Python 3.14+ a instalação
# falha com "No matching distribution found for torch". Detecta cedo e
# orienta a recriar o venv com uma versão suportada.
PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYMINOR="$(python -c 'import sys; print(sys.version_info.minor)')"
echo "==> Python do venv: ${PYVER}"
if [[ "${PYMINOR}" -gt 13 ]]; then
    echo "ERRO: Python ${PYVER} ainda não é suportado pelos wheels do PyTorch"
    echo "      (suportado: 3.9–3.13). Recrie o venv com um Python suportado:"
    echo ""
    exit 1
fi

echo "==> Atualizando pip..."
python -m pip install --upgrade pip >/dev/null

# --- aviso sobre driver -----------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> nvidia-smi detectado:"
    nvidia-smi -L || true
else
    echo "AVISO: 'nvidia-smi' não encontrado no PATH."
    echo "       Os wheels do PyTorch trazem as libs CUDA, mas você ainda"
    echo "       precisa do DRIVER NVIDIA instalado. No Fedora:"
    echo "         sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda"
fi

# --- PyTorch (CUDA) ---------------------------------------------------------
# Reinstala para garantir que a build CUDA substitua qualquer torch CPU-only.
echo "==> Instalando PyTorch (CUDA ${CUDA_CHANNEL})..."
pip install --upgrade --force-reinstall \
    torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"

# --- dependências de fine-tuning -------------------------------------------
echo "==> Instalando dependências de fine-tuning..."
pip install transformers peft datasets accelerate sentencepiece sacremoses

# --- verificação ------------------------------------------------------------
echo "==> Verificando a instalação..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("bf16 suportado:", torch.cuda.is_bf16_supported())
else:
    print("ATENCAO: CUDA indisponivel. Verifique o driver NVIDIA e o canal CUDA.")
PY

echo ""
echo "==> Pronto. Para treinar com LLaMA na GPU:"
echo "    source \"${VENV_DIR}/bin/activate\""
echo "    python -m src.finetuning.train --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --epochs 3"
