#!/usr/bin/env bash
#
# Instala o PyTorch com suporte a ROCm (GPUs AMD) + dependências de
# fine-tuning, dentro do virtualenv do projeto.
#
# ROCm é suportado apenas em Linux. Verifique se sua GPU AMD está na lista
# de suporte do ROCm antes de usar: https://rocm.docs.amd.com
#
# Uso:
#   ./scripts/install_torch_amd.sh
#
# Variáveis de ambiente opcionais:
#   ROCM_CHANNEL   Canal de wheels do PyTorch (padrão: rocm6.1).
#                  Ex.: rocm6.0, rocm6.1, rocm6.2 — veja https://pytorch.org
#   VENV_DIR       Caminho do virtualenv (padrão: venv na raiz do projeto).
#
set -euo pipefail

# Raiz do projeto = diretório pai da pasta deste script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ROCM_CHANNEL="${ROCM_CHANNEL:-rocm6.1}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/venv}"

echo "==> Projeto:        ${PROJECT_ROOT}"
echo "==> Virtualenv:     ${VENV_DIR}"
echo "==> Canal ROCm:     ${ROCM_CHANNEL}"

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

# --- aviso sobre driver/ROCm ------------------------------------------------
if command -v rocminfo >/dev/null 2>&1; then
    echo "==> ROCm detectado:"
    rocminfo | grep -i "Marketing Name" || true
else
    echo "AVISO: 'rocminfo' não encontrado no PATH."
    echo "       Os wheels do PyTorch trazem as libs ROCm, mas você ainda"
    echo "       precisa do STACK ROCm/driver amdgpu instalado no sistema."
    echo "       Veja: https://rocm.docs.amd.com/projects/install-on-linux/"
fi

# --- PyTorch (ROCm) ---------------------------------------------------------
# Reinstala para garantir que a build ROCm substitua qualquer torch
# CPU-only ou CUDA previamente instalado.
echo "==> Instalando PyTorch (ROCm ${ROCM_CHANNEL})..."
pip install --upgrade --force-reinstall \
    torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${ROCM_CHANNEL}"

# --- dependências de fine-tuning -------------------------------------------
echo "==> Instalando dependências de fine-tuning..."
pip install transformers peft datasets accelerate sentencepiece sacremoses

# --- verificação ------------------------------------------------------------
# No PyTorch com ROCm, a API usada é a mesma `torch.cuda` (HIP emula CUDA),
# então torch.cuda.is_available() == True e torch.version.hip fica populado.
echo "==> Verificando a instalação..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("ROCm/HIP:", getattr(torch.version, "hip", None))
print("GPU disponível (via torch.cuda):", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    try:
        print("bf16 suportado:", torch.cuda.is_bf16_supported())
    except Exception as exc:
        print("bf16 suportado: indeterminado (", exc, ")")
else:
    print("ATENCAO: GPU indisponivel. Verifique o stack ROCm e o canal escolhido.")
PY

echo ""
echo "==> Pronto. Para treinar com LLaMA na GPU:"
echo "    source \"${VENV_DIR}/bin/activate\""
echo "    python -m src.finetuning.train --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --epochs 3"
echo ""
echo "Dica AMD: se sua GPU não for oficialmente suportada, pode ser"
echo "necessário exportar HSA_OVERRIDE_GFX_VERSION antes de treinar."
