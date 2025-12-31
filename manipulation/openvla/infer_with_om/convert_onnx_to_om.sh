#!/bin/bash
# Convert OpenVLA ONNX models to Ascend OM format
#
# Usage:
#   ./convert_onnx_to_om.sh \
#     --vision-onnx-dir <vision_onnx_dir> \
#     --llama-prefill-onnx-dir <prefill_onnx_dir> \
#     --llama-decoder-onnx-dir <decoder_onnx_dir> \
#     --vision-om-dir <vision_om_dir> \
#     --llama-prefill-om-dir <prefill_om_dir> \
#     --llama-decoder-om-dir <decoder_om_dir> \
#     [--soc-version <soc_version>]

set -e  # Exit on error

# Default values
SOC_VERSION="Ascend310P3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vision-onnx-dir)
            VISION_ONNX_DIR="$2"
            shift 2
            ;;
        --llama-prefill-onnx-dir)
            LLAMA_PREFILL_ONNX_DIR="$2"
            shift 2
            ;;
        --llama-decoder-onnx-dir)
            LLAMA_DECODER_ONNX_DIR="$2"
            shift 2
            ;;
        --vision-om-dir)
            VISION_OM_DIR="$2"
            shift 2
            ;;
        --llama-prefill-om-dir)
            LLAMA_PREFILL_OM_DIR="$2"
            shift 2
            ;;
        --llama-decoder-om-dir)
            LLAMA_DECODER_OM_DIR="$2"
            shift 2
            ;;
        --soc-version)
            SOC_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --vision-onnx-dir DIR         Directory containing vision ONNX models"
            echo "  --llama-prefill-onnx-dir DIR  Directory containing prefill ONNX model"
            echo "  --llama-decoder-onnx-dir DIR  Directory containing decoder ONNX model"
            echo "  --vision-om-dir DIR           Output directory for vision OM models"
            echo "  --llama-prefill-om-dir DIR    Output directory for prefill OM model"
            echo "  --llama-decoder-om-dir DIR    Output directory for decoder OM model"
            echo "  --soc-version VERSION         SOC version (default: Ascend310P3)"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 \\"
            echo "    --vision-onnx-dir outputs/onnx/vision \\"
            echo "    --llama-prefill-onnx-dir outputs/onnx/llama_prefill \\"
            echo "    --llama-decoder-onnx-dir outputs/onnx/llama_decoder \\"
            echo "    --vision-om-dir outputs/om/vision \\"
            echo "    --llama-prefill-om-dir outputs/om/llama_prefill \\"
            echo "    --llama-decoder-om-dir outputs/om/llama_decoder"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$VISION_ONNX_DIR" ]] || [[ -z "$LLAMA_PREFILL_ONNX_DIR" ]] || [[ -z "$LLAMA_DECODER_ONNX_DIR" ]] || \
   [[ -z "$VISION_OM_DIR" ]] || [[ -z "$LLAMA_PREFILL_OM_DIR" ]] || [[ -z "$LLAMA_DECODER_OM_DIR" ]]; then
    echo "Error: All directory arguments are required"
    echo "Use --help for usage information"
    exit 1
fi

# Function to convert ONNX to OM
# Args:
#   1) onnx_path
#   2) om_output_dir
#   3) model_name
#   4+) extra atc args (optional)
convert_onnx_to_om() {
    local onnx_path="$1"
    local om_output_dir="$2"
    local model_name="$3"
    shift 3
    local extra_args=("$@")

    if [[ ! -f "$onnx_path" ]]; then
        echo "Error: ONNX model not found: $onnx_path"
        return 1
    fi

    echo "=========================================="
    echo "Converting: $model_name"
    echo "  Input:  $onnx_path"
    echo "  Output: $om_output_dir/$model_name"
    echo "=========================================="

    mkdir -p "$om_output_dir"

    # Run atc inside ONNX directory so external weights can be resolved correctly
    local abs_onnx abs_out onnx_dir onnx_file
    abs_onnx="$(realpath "$onnx_path")"
    abs_out="$(realpath "$om_output_dir")"
    onnx_dir="$(dirname "$abs_onnx")"
    onnx_file="$(basename "$abs_onnx")"

    pushd "$onnx_dir" >/dev/null

    # Use relative model path (./model.onnx) under ONNX dir, but absolute output path
    atc --model="./$onnx_file" \
        --framework=5 \
        --output="$abs_out/$model_name" \
        --soc_version="$SOC_VERSION" \
        "${extra_args[@]}"

    popd >/dev/null

    echo "✓ Successfully converted: $model_name"
    echo ""
}

# Convert Vision models
echo "=========================================="
echo "Converting Vision Models"
echo "=========================================="
echo ""

# Note: vision_* 不加 --precision_mode_v2=origin
convert_onnx_to_om \
    "$VISION_ONNX_DIR/vision_backbone.onnx" \
    "$VISION_OM_DIR" \
    "vision_backbone"

convert_onnx_to_om \
    "$VISION_ONNX_DIR/projector.onnx" \
    "$VISION_OM_DIR" \
    "projector"

# embedding 需要额外参数：input_shape / dynamic_dims / input_format
convert_onnx_to_om \
    "$VISION_ONNX_DIR/embedding.onnx" \
    "$VISION_OM_DIR" \
    "embedding" \
    --input_shape="input_ids:1,-1" \
    --dynamic_dims="1;32" \
    --input_format=ND

# Convert LLaMA Prefill model
echo "=========================================="
echo "Converting LLaMA Prefill Model"
echo "=========================================="
echo ""

# prefill 才加 precision_mode_v2=origin
convert_onnx_to_om \
    "$LLAMA_PREFILL_ONNX_DIR/vla_prefill.onnx" \
    "$LLAMA_PREFILL_OM_DIR" \
    "vla_prefill" \
    --precision_mode_v2=origin

# Convert LLaMA Decoder model
echo "=========================================="
echo "Converting LLaMA Decoder Model"
echo "=========================================="
echo ""

# decoder 才加 precision_mode_v2=origin，并加入 input_shape="input_ids:1,1"
convert_onnx_to_om \
    "$LLAMA_DECODER_ONNX_DIR/vla_decoder.onnx" \
    "$LLAMA_DECODER_OM_DIR" \
    "vla_decoder" \
    --precision_mode_v2=origin \
    --input_shape="input_ids:1,1"

echo "=========================================="
echo "All conversions completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Vision models:      $VISION_OM_DIR"
echo "  Prefill model:      $LLAMA_PREFILL_OM_DIR"
echo "  Decoder model:      $LLAMA_DECODER_OM_DIR"
echo "  SOC Version:        $SOC_VERSION"
echo ""
