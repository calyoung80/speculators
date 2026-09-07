#!/bin/bash
# Copy only the BF16 verifier files DFlash2 training needs on the Consumer node.
set -euo pipefail

PASS=${D2_SSH_PASSWORD:?Set D2_SSH_PASSWORD before running this script}
USER=test_mtp
SOURCE_HOST=71.10.29.119
DESTINATION_HOST=71.10.29.119
SOURCE_PATH=/mnt/share/weight
DESTINATION_PATH=/mnt/hcs/y00917737/dflash2_verifier_minimal
STAGING_PATH=${DESTINATION_PATH}.partial

ssh_cmd=(
  sshpass -p "${PASS}" ssh
  -o PreferredAuthentications=password
  -o PubkeyAuthentication=no
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=15
)
manifest=$(mktemp)
trap 'rm -f "${manifest}"' EXIT

"${ssh_cmd[@]}" "${USER}@${SOURCE_HOST}" \
  "echo '${PASS}' | sudo -S python3 -c 'import json; from pathlib import Path; from safetensors import safe_open; root=Path(\"${SOURCE_PATH}\"); index_path=root / \"model.safetensors.index.json\"; weight_map=json.loads(index_path.read_text())[\"weight_map\"] if index_path.exists() else {}; required={\"embed_tokens.weight\": (\"embed_tokens.weight\", \"tok_embeddings.weight\", \"llm.embed.weight\"), \"lm_head.weight\": (\"lm_head.weight\", \"output.weight\", \"llm.unembed.weight\"), \"model.norm.weight\": (\"llm.norm.weight\", \"norm.weight\")};
if not weight_map:
  for shard in root.glob(\"*.safetensors\"):
    with safe_open(str(shard), framework=\"pt\") as handle:
      for key in handle.keys(): weight_map[key]=shard.name
selected={}
for name, aliases in required.items():
  key=None
  for alias in (name, *aliases):
    matches=[candidate for candidate in weight_map if candidate == alias or candidate.endswith(alias)]
    if matches: key=min(matches, key=len); break
  if key is None: raise ValueError(f\"missing {name}; root files={[path.name for path in root.iterdir()]}; candidate keys={list(weight_map)[:20]}\")
  selected[key]=weight_map[key]
print(json.dumps({\"files\": sorted(set(selected.values())), \"weight_map\": selected}))'" \
  > "${manifest}"

"${ssh_cmd[@]}" "${USER}@${DESTINATION_HOST}" \
  "echo '${PASS}' | sudo -S sh -c 'test ! -e ${DESTINATION_PATH}; test ! -e ${STAGING_PATH}; mkdir ${STAGING_PATH}'"

copy_file() {
  local file=$1
  "${ssh_cmd[@]}" "${USER}@${DESTINATION_HOST}" \
    "echo '${PASS}' | sudo -S cp --reflink=auto --sparse=always '${SOURCE_PATH}/${file}' '${STAGING_PATH}/${file}'" \
    </dev/null
}

while IFS= read -r file; do
  copy_file "${file}"
done < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["files"]))' "${manifest}")

python3 -c 'import json,sys; print(json.dumps({"weight_map": json.load(open(sys.argv[1]))["weight_map"]}))' "${manifest}" \
  | "${ssh_cmd[@]}" "${USER}@${DESTINATION_HOST}" \
    "{ printf '%s\\n' '${PASS}'; cat; } | sudo -S tee '${STAGING_PATH}/model.safetensors.index.json' >/dev/null"

metadata_files=(
  chat_template.jinja
  config.json
  configuration.json
  generation_config.json
  merges.txt
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  video_preprocessor_config.json
  vocab.json
)
for file in "${metadata_files[@]}"; do
  if "${ssh_cmd[@]}" "${USER}@${SOURCE_HOST}" \
    "echo '${PASS}' | sudo -S test -f '${SOURCE_PATH}/${file}'"; then
    copy_file "${file}"
  fi
done

"${ssh_cmd[@]}" "${USER}@${DESTINATION_HOST}" \
  "echo '${PASS}' | sudo -S mv ${STAGING_PATH} ${DESTINATION_PATH}; docker exec dflash2_train python3 -c 'from pathlib import Path; from transformers import AutoConfig, AutoTokenizer; root=Path(\"${DESTINATION_PATH}\"); assert (root / \"model.safetensors.index.json\").is_file(); AutoConfig.from_pretrained(root); AutoTokenizer.from_pretrained(root); print(root)'"
