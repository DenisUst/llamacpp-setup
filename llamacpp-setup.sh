#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/var/log/llamacpp-setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Параметры (переопределяются при запуске: INSTALL_DIR=/opt/llama.cpp ./llamacpp-setup.sh) ---
INSTALL_DIR="${INSTALL_DIR:-/opt/llama.cpp}"
CUDA_ARCH="${CUDA_ARCH:-}"   # например "86" для RTX 3080/3090, "89" для 4090. Пусто = автоопределение сборкой.
MODEL_URL="${MODEL_URL:-}"  # прямая ссылка на .gguf файл (например с HuggingFace)
MODELS_DIR="${MODELS_DIR:-${INSTALL_DIR}/models}"
SERVER_PORT="${SERVER_PORT:-8080}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
NGL="${NGL:-99}"   # количество слоёв на GPU, 99 = все слои
START_SERVER="${START_SERVER:-false}"  # true = запустить llama-server как systemd-службу

echo "=== llama.cpp setup started: $(date) ==="

# На некоторых образах sudo не подхватывает /etc/profile.d/*, поэтому
# подстраховываемся явным добавлением CUDA в PATH для этого запуска скрипта.
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "Запусти скрипт от root (sudo)."
        exit 1
    fi
}

# 1. Проверка GPU/CUDA
verify_gpu() {
    echo "--- [1/7] Проверка GPU/CUDA ---"
    if ! command -v nvidia-smi &>/dev/null; then
        echo "nvidia-smi не найден — без GPU-драйвера CUDA-сборка не соберётся. Прерываю."
        exit 1
    fi
    nvidia-smi
    if ! command -v nvcc &>/dev/null; then
        echo "nvcc не найден — убедись, что CUDA Toolkit установлен и добавлен в PATH."
        exit 1
    fi
    nvcc --version
}

# 2. Установка зависимостей сборки
install_build_deps() {
    echo "--- [2/7] Установка зависимостей сборки ---"
    apt-get update -y
    apt-get install -y build-essential cmake git libcurl4-openssl-dev
}

# 3. Клонирование репозитория
clone_repo() {
    echo "--- [3/7] Клонирование llama.cpp ---"
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        echo "Репозиторий уже существует в ${INSTALL_DIR}, обновляю."
        git -C "$INSTALL_DIR" pull
    else
        git clone https://github.com/ggml-org/llama.cpp.git "$INSTALL_DIR"
    fi
}

# 4. Сборка с поддержкой CUDA
build_llamacpp() {
    echo "--- [4/7] Сборка с CUDA ---"
    local build_dir="${INSTALL_DIR}/build"
    local cmake_args=(-B "$build_dir" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release)

    if [[ -n "$CUDA_ARCH" ]]; then
        cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
    fi

    cmake -S "$INSTALL_DIR" "${cmake_args[@]}"
    cmake --build "$build_dir" --config Release -j"$(nproc)"

    echo "Сборка завершена. Бинарники в ${build_dir}/bin"
}

# 5. Скачивание модели по ссылке
download_model() {
    echo "--- [5/7] Скачивание модели ---"
    if [[ -z "$MODEL_URL" ]]; then
        echo "Переменная MODEL_URL не задана — пропуск загрузки модели."
        echo "Пример запуска: MODEL_URL=https://huggingface.co/.../model.gguf ./llamacpp-setup.sh"
        return
    fi

    mkdir -p "$MODELS_DIR"
    local filename
    filename="$(basename "${MODEL_URL%%\?*}")"   # отбрасываем query-параметры из URL (частое дело у HF-ссылок)
    local target_path="${MODELS_DIR}/${filename}"

    if [[ -f "$target_path" ]]; then
        echo "Файл ${target_path} уже существует, пропуск скачивания."
        echo "Чтобы перекачать — удали файл вручную и перезапусти скрипт."
    else
        echo "Скачиваю ${MODEL_URL} -> ${target_path}"
        curl -fL --retry 3 -o "$target_path" "$MODEL_URL"
        echo "Модель сохранена: ${target_path}"
    fi

    echo "Для запуска: ${INSTALL_DIR}/build/bin/llama-cli -m ${target_path} -ngl 99 -p \"Привет\""
}

# 6. Проверка сборки
verify_build() {
    echo "--- [6/7] Проверка сборки ---"
    local cli_bin="${INSTALL_DIR}/build/bin/llama-cli"
    if [[ -x "$cli_bin" ]]; then
        "$cli_bin" --version
        echo "llama.cpp собран успешно: ${cli_bin}"
    else
        echo "Бинарник llama-cli не найден — сборка не завершилась корректно."
        exit 1
    fi
}

# 7. Настройка llama-server как systemd-службы
setup_server_service() {
    echo "--- [7/7] Настройка llama-server ---"

    if [[ "$START_SERVER" != "true" ]]; then
        echo "START_SERVER не установлен в true — пропуск настройки службы."
        echo "Для запуска вручную: ${INSTALL_DIR}/build/bin/llama-server -m <путь к .gguf> --host ${SERVER_HOST} --port ${SERVER_PORT} -ngl ${NGL}"
        return
    fi

    if [[ -z "$MODEL_URL" ]]; then
        echo "START_SERVER=true, но MODEL_URL не задан — нечего загружать в сервер. Пропуск."
        return
    fi

    local server_bin="${INSTALL_DIR}/build/bin/llama-server"
    local filename
    filename="$(basename "${MODEL_URL%%\?*}")"
    local model_path="${MODELS_DIR}/${filename}"

    if [[ ! -x "$server_bin" ]]; then
        echo "Бинарник llama-server не найден по пути ${server_bin}. Пропуск."
        return
    fi
    if [[ ! -f "$model_path" ]]; then
        echo "Файл модели ${model_path} не найден. Пропуск."
        return
    fi

    cat > /etc/systemd/system/llama-server.service <<EOF
[Unit]
Description=llama.cpp server
After=network.target

[Service]
Type=simple
ExecStart=${server_bin} -m ${model_path} --host ${SERVER_HOST} --port ${SERVER_PORT} -ngl ${NGL}
Restart=on-failure
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable llama-server
    systemctl restart llama-server

    sleep 2
    if systemctl is-active --quiet llama-server; then
        echo "llama-server запущен и слушает на ${SERVER_HOST}:${SERVER_PORT}"
        echo "Проверка: curl http://127.0.0.1:${SERVER_PORT}/health"
    else
        echo "llama-server не запустился. Проверь: journalctl -u llama-server -n 50"
        exit 1
    fi
}

main() {
    check_root
    verify_gpu
    install_build_deps
    clone_repo
    build_llamacpp
    download_model
    verify_build
    setup_server_service
    echo "=== Setup завершён: $(date) ==="
    echo ""
    if [[ -z "$MODEL_URL" ]]; then
        echo "Модель не скачана (MODEL_URL не задан). Пример запуска вручную:"
        echo "  ${INSTALL_DIR}/build/bin/llama-cli -m /path/to/model.gguf -ngl 99 -p \"Привет\""
    fi
    echo "-ngl 99 — выгрузить все слои на GPU"
}

main "$@"
