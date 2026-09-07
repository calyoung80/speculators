#!/bin/bash
# Copy the full Qwen3.8-27B W8A8 model from 119 to 118 shared storage.
set -euo pipefail

PASS=${D2_SSH_PASSWORD:?Set D2_SSH_PASSWORD before running this script}
USER=test_mtp
SOURCE_HOST=71.10.29.119
DESTINATION_HOST=71.10.29.118
SOURCE_PATH=/mnt/hcs/weights/Qwen3.8-27B-w8a8
DESTINATION_PATH=/mnt/share/weight/Qwen3.8-27B-w8a8
STAGING_PATH=${DESTINATION_PATH}.partial

ssh_cmd=(
  sshpass -p "${PASS}" ssh
  -o PreferredAuthentications=password
  -o PubkeyAuthentication=no
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=30
)

MODE=${1:-copy}

if [ "${MODE}" = cleanup ]; then
  "${ssh_cmd[@]}" "${USER}@${DESTINATION_HOST}" \
    "echo '${PASS}' | sudo -S sh -c 'for path in ${DESTINATION_PATH} ${STAGING_PATH}; do test ! -e \"\$path\" || rmdir \"\$path\"; done'"
  exit 0
fi

if [ "${MODE}" != copy ] && [ "${MODE}" != publish-staged ]; then
  echo "Usage: $0 {copy|publish-staged|cleanup}" >&2
  exit 2
fi

source_info=$("${ssh_cmd[@]}" "${USER}@${SOURCE_HOST}" \
  "echo '${PASS}' | sudo -S sh -c 'test -f ${SOURCE_PATH}/config.json && du -sb ${SOURCE_PATH} && sha256sum ${SOURCE_PATH}/config.json && find ${SOURCE_PATH} -type f | wc -l'")

source_size=$(printf '%s\n' "${source_info}" | sed -n '1s/\t.*//p')
source_hash=$(printf '%s\n' "${source_info}" | sed -n '2s/ .*//p')
source_files=$(printf '%s\n' "${source_info}" | sed -n '3p')

if [ -z "${source_size}" ] || [ -z "${source_hash}" ] || [ -z "${source_files}" ]; then
  echo "Source model is unavailable: ${SOURCE_HOST}:${SOURCE_PATH}" >&2
  exit 1
fi

if [ "${MODE}" = copy ]; then
  printf '%s\nstream-probe\n' "${PASS}" \
    | "${ssh_cmd[@]}" "${USER}@${DESTINATION_HOST}" \
      "sudo -S sh -c 'IFS= read -r marker; test \"\$marker\" = stream-probe'"

  "${ssh_cmd[@]}" "${USER}@${DESTINATION_HOST}" \
    "echo '${PASS}' | sudo -S sh -c 'test ! -e ${DESTINATION_PATH}; test ! -e ${STAGING_PATH}; available=\$(df -PB1 /mnt/share/weight | awk \"END {print \\\$4}\"); test \"\$available\" -gt ${source_size}'"

  "${ssh_cmd[@]}" "${USER}@${SOURCE_HOST}" \
    "echo '${PASS}' | sudo -S tar -C ${SOURCE_PATH} -cf - ." \
    | "${ssh_cmd[@]}" "${USER}@${DESTINATION_HOST}" \
      "{ printf '%s\\n' '${PASS}'; cat; } | sudo -S sh -c 'mkdir ${STAGING_PATH} && exec tar --no-same-owner --no-same-permissions -C ${STAGING_PATH} -xf -'"
fi

destination_info=$("${ssh_cmd[@]}" "${USER}@${DESTINATION_HOST}" \
  "echo '${PASS}' | sudo -S sh -c 'sha256sum ${STAGING_PATH}/config.json && find ${STAGING_PATH} -type f | wc -l'")
destination_hash=$(printf '%s\n' "${destination_info}" | sed -n '1s/ .*//p')
destination_files=$(printf '%s\n' "${destination_info}" | sed -n '2p')

if [ "${source_hash}" != "${destination_hash}" ] || [ "${source_files}" != "${destination_files}" ]; then
  echo "Model copy validation failed; staging directory retained at ${STAGING_PATH}." >&2
  exit 1
fi

"${ssh_cmd[@]}" "${USER}@${DESTINATION_HOST}" \
  "echo '${PASS}' | sudo -S mv ${STAGING_PATH} ${DESTINATION_PATH}"

printf 'Copied %s files (%s bytes) to %s:%s\n' \
  "${source_files}" "${source_size}" "${DESTINATION_HOST}" "${DESTINATION_PATH}"
