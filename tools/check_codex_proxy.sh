#!/usr/bin/env bash
# Diagnose Codex <-> chatgpt.com connectivity on this Linux host.
set -euo pipefail

echo "=== 1) local :7890 or 17890 (tunnel) ==="
if (echo >/dev/tcp/127.0.0.1/7890) >/dev/null 2>&1; then
  echo "OK  127.0.0.1:7890 is open"
else
  echo "FAIL 127.0.0.1:7890 closed — start Windows Clash + SSH reverse tunnel:"
  echo "  # on Windows PowerShell / terminal (Clash must listen on 7890):"
  echo "  ssh -N -R 7890:127.0.0.1:7890 USER@LINUX_HOST"
  echo "  # or add to Windows ~/.ssh/config Host entry:"
  echo "  #   RemoteForward 7890 127.0.0.1:7890"
fi

echo
echo "=== 2) proxy env ==="
echo "HTTP_PROXY=${HTTP_PROXY:-<unset>} HTTPS_PROXY=${HTTPS_PROXY:-<unset>}"

echo
echo "=== 3) reachability ==="
unset_proxy() { unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; }
try() {
  local label=$1; shift
  local code
  code=$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 6 --max-time 10 "$@" 2>/dev/null || echo FAIL)
  echo "$label -> $code"
}

echo "-- with current env --"
try "chatgpt.com" https://chatgpt.com/
try "api.openai.com" https://api.openai.com/v1/models

echo "-- forced via 127.0.0.1:7890 --"
try "chatgpt via proxy" -x http://127.0.0.1:7890 https://chatgpt.com/

echo "-- direct no proxy --"
unset_proxy
try "chatgpt direct" https://chatgpt.com/
try "baidu direct" https://www.baidu.com/

echo
echo "=== 4) how to fix Codex plugin ==="
echo "1. Windows: start Clash/V2Ray, mixed/HTTP port = 7890"
echo "2. Windows: keep an SSH reverse tunnel open (see above)"
echo "3. Linux: bash $0   # expect chatgpt via proxy -> 200/301/302"
echo "4. Cursor: Reload Window / reconnect Remote-SSH, reopen Codex"
echo "5. ChatGPT login in Codex must still be valid"
