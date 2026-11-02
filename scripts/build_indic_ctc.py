#!/usr/bin/env python3
"""Build the exact Indic encoder plus its language-selected CTC head."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/models/indicconformer")
    p.add_argument("--output", required=True)
    p.add_argument("--language", default="hi")
    p.add_argument("--precision", choices=["fp32"], default="fp32",
        help="FP32 weights with TF32 tensor-core operations allowed.")
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--head-only", action="store_true")
    args = p.parse_args()
    source = Path(args.model) / "assets"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    masks = json.loads((source / "language_masks.json").read_text())
    selected = np.asarray(masks[args.language], dtype=bool)
    head = onnx.load(source / "ctc_decoder.onnx")
    weights = {tensor.name: numpy_helper.to_array(tensor) for tensor in head.graph.initializer}
    weight = weights["language_ids"][selected]
    bias = weights["decoder_layers.0.bias"][selected]
    initializers = [numpy_helper.from_array(weight, "indic_ctc_weight"), numpy_helper.from_array(bias, "indic_ctc_bias")]
    nodes = [helper.make_node("Conv", ["encoder_output", "indic_ctc_weight", "indic_ctc_bias"], ["indic_ctc_conv"]),
             helper.make_node("Transpose", ["indic_ctc_conv"], ["logprobs"], perm=[0, 2, 1])]
    output = helper.make_tensor_value_info("logprobs", onnx.TensorProto.FLOAT, ["batch", "time", int(selected.sum())])
    graph = helper.make_graph(nodes, "Indic language-selected CTC", [head.graph.input[0]], [output], initializers)
    pruned = helper.make_model(graph, opset_imports=head.opset_import, ir_version=head.ir_version)
    onnx.save(pruned, out / "ctc_head.onnx")
    (out / "settings.json").write_text(json.dumps(vars(args), indent=2))
    if args.head_only:
        return
    # Preserve the official encoder weights and FP32 computation graph.
    model = onnx.load(source / "encoder.onnx", load_external_data=True)
    for node in nodes:
        if node.op_type == "Conv":
            node.input[0] = "outputs"
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(initializers)
    encoded_lengths = next(x for x in model.graph.output if x.name == "encoded_lengths")
    del model.graph.output[:]
    model.graph.output.extend([output, encoded_lengths])
    weight_path = out / "weights.bin"
    weight_path.unlink(missing_ok=True)
    onnx.save_model(model, out / "model.onnx", save_as_external_data=True,
        all_tensors_to_one_file=True, location="weights.bin", size_threshold=1024)
    del model
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(out / "model.onnx")):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    config.set_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape("audio_signal", (1,80,16), (min(32,args.max_batch),80,1000), (args.max_batch,80,2016))
    profile.set_shape("length", (1,), (min(32,args.max_batch),), (args.max_batch,))
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")
    (out / "model.plan").write_bytes(bytes(engine))
    print("BUILT",out / "model.plan",flush=True)


if __name__ == "__main__":
    main()
