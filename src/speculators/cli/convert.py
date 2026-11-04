"""Convert command — convert external models to speculators format."""

import json
from enum import StrEnum
from typing import Annotated, Any

import typer

from speculators.convert import convert_model


class AlgorithmChoice(StrEnum):
    eagle3 = "eagle3"
    mtp = "mtp"
    dflash = "dflash"


def convert(
    model: Annotated[
        str,
        typer.Argument(help="Model checkpoint or Hugging Face model ID to convert"),
    ],
    verifier: Annotated[
        str,
        typer.Option(
            "--verifier",
            help=(
                "Verifier model checkpoint or Hugging Face model ID "
                "to attach as the verification/base model for speculative decoding"
            ),
        ),
    ],
    algorithm: Annotated[
        AlgorithmChoice,
        typer.Option(
            help=(
                "The source repo/algorithm to convert from into the matching "
                "algorithm in Speculators"
            ),
        ),
    ],
    output_path: Annotated[
        str, typer.Option(help="Directory path where converted model will be saved")
    ] = "converted",
    validate_device: Annotated[
        str | None,
        typer.Option(
            help=(
                "Device to validate the model on (e.g. 'cuda:0') "
                "If not provided, validation is skipped."
            ),
        ),
    ] = None,
    native_mtp_key_prefix: Annotated[
        str | None,
        typer.Option(
            "--native-mtp-key-prefix",
            help="Native GLM MTP layer prefix, e.g. 'model.layers.78.'.",
        ),
    ] = None,
    index_name: Annotated[
        str | None,
        typer.Option(
            "--index-name",
            help=(
                "Safetensors index filename, e.g. "
                "'quant_model_weights.safetensors.index.json'."
            ),
        ),
    ] = None,
    own_mtp_head: Annotated[
        bool | None,
        typer.Option(
            "--own-mtp-head/--no-own-mtp-head",
            help="Native checkpoint ships its own MTP LM head.",
        ),
    ] = None,
    own_mtp_embed: Annotated[
        bool | None,
        typer.Option(
            "--own-mtp-embed/--no-own-mtp-embed",
            help="Native checkpoint ships its own MTP embedding.",
        ),
    ] = None,
    has_rot_weight: Annotated[
        bool | None,
        typer.Option(
            "--has-rot-weight/--no-has-rot-weight",
            help="Native checkpoint ships model.layers.<N>.rot.weight.",
        ),
    ] = None,
    is_rot_used: Annotated[
        bool | None,
        typer.Option(
            "--is-rot-used/--no-is-rot-used",
            help=(
                "Quant metadata claims the rot projection is used; must agree "
                "with --has-rot-weight (rot_gate_consistent)."
            ),
        ),
    ] = None,
    quant_metadata_policy: Annotated[
        str | None,
        typer.Option(
            "--quant-metadata-policy",
            help="Quant metadata policy; only 'copy-rebuild' is supported.",
        ),
    ] = None,
    key_map_out: Annotated[
        str | None,
        typer.Option(
            "--key-map-out",
            help="Write the GLM MTP native key map to this path.",
        ),
    ] = None,
    algorithm_kwargs: Annotated[
        dict[str, Any] | None,
        typer.Option(
            parser=json.loads,
            help=(
                "Additional keyword args for the conversion alg as a JSON string. "
                'Options for Eagle3: {"norm_before_residual": true, '
                '"eagle_aux_hidden_state_layer_ids": [1,23,44]}. '
                'Options for MTP: {"num_speculative_steps": 3}. '
                'Options for DFlash: {"aux_hidden_state_layer_ids": [2,10,18,26,34]}.'
            ),
        ),
    ] = None,
):
    """Convert models from external research repositories or formats
    into the standardized Speculators format for use within the Speculators
    framework, Hugging Face model hub compatibility, and deployment with vLLM.
    Supported algorithms, repositories, and examples given below.

    \b
    algorithm=="eagle3":
        Eagle v3: https://github.com/SafeAILab/EAGLE
        ::
        # general
        speculators convert "./eagle/checkpoint" \\
            --algorithm eagle3 \\
            --verifier "meta-llama/Llama-3.1-8B-Instruct"
        # with normalization before the residual
        speculators convert "./eagle/checkpoint" \\
            --algorithm eagle3 \\
            --algorithm-kwargs '{"norm_before_residual": true}' \\
            --verifier "meta-llama/Llama-3.1-8B-Instruct"

    \b
    algorithm=="mtp":
        MTP (Multi-Token Prediction): models with native MTP layers
        (e.g. Qwen3-Next, Qwen3.5, Qwen3.5-MoE, GLM-MoE-DSA)
        ::
        speculators convert "Qwen/Qwen3-Next-80B-A3B-Instruct" \\
            --algorithm mtp \\
            --verifier "Qwen/Qwen3-Next-80B-A3B-Instruct" \\
            --algorithm-kwargs '{"num_speculative_steps": 3}'
        # GLM-MoE-DSA native head
        speculators convert "/path/GLM-5.3" \\
            --algorithm mtp \\
            --verifier "/path/GLM-5.3" \\
            --native-mtp-key-prefix "model.layers.78." \\
            --index-name "quant_model_weights.safetensors.index.json" \\
            --no-own-mtp-head --no-own-mtp-embed --no-has-rot-weight

    \b
    algorithm=="dflash":
        DFlash: https://z-lab.ai/projects/dflash/
        ::
        speculators convert "z-lab/Qwen3-8B-DFlash-b16" \\
            --algorithm dflash \\
            --verifier "Qwen/Qwen3-8B"
    """
    if not algorithm_kwargs:
        algorithm_kwargs = {}
    elif not isinstance(algorithm_kwargs, dict):
        raise typer.BadParameter(
            "--algorithm-kwargs must be a JSON object, not "
            + type(algorithm_kwargs).__name__
        )

    if algorithm == AlgorithmChoice.mtp:
        glm_options = {
            "native_mtp_key_prefix": native_mtp_key_prefix,
            "index_name": index_name,
            "own_mtp_head": own_mtp_head,
            "own_mtp_embed": own_mtp_embed,
            "has_rot_weight": has_rot_weight,
            "is_rot_used": is_rot_used,
            "quant_metadata_policy": quant_metadata_policy,
            "key_map_out": key_map_out,
        }
        algorithm_kwargs.update(
            {key: value for key, value in glm_options.items() if value is not None}
        )

    convert_model(
        model=model,
        verifier=verifier,
        output_path=output_path,
        validate_device=validate_device,
        algorithm=algorithm.value,
        **algorithm_kwargs,
    )
