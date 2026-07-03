import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from onnxslim import slim
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torch.export import Dim

from ultralytics import YOLO
from ultralytics.utils.ax_quantizer import AXQuantizer, ax_load_config

warnings.filterwarnings("ignore", message=r"erase_node\(batch_norm_.*")


@dataclass(frozen=True)
class ExportDefaults:
    task: str
    model: str
    pretrained: str
    qat_weights: str
    out: str
    qat_state_out: str


DEFAULT_EXPORTS = {
    "detect": ExportDefaults(
        task="detect",
        model="yolo26n.yaml",
        pretrained="yolo26n.pt",
        qat_weights="runs/detect/qat/weights/best.pt",
        out="./runs/qat_one2many.onnx",
        qat_state_out="./runs/qat_one2many.pth",
    ),
    "segment": ExportDefaults(
        task="segment",
        model="yolo26n-seg.yaml",
        pretrained="./weights/yolo26n-seg.pt",
        qat_weights="runs/segment/qat/weights/best.pt",
        out="./runs/qat-seg.onnx",
        qat_state_out="./runs/qat-seg.pth",
    ),
}

REQUIRED_ONNX_OPSET = 21  # int16 Q/DQ 需要更高 opset，ORT 在 opset 18 下会拒绝加载


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO26 PT2E QAT model to ONNX.")
    parser.add_argument("--task", choices=sorted(DEFAULT_EXPORTS), default="detect")
    parser.add_argument("--model", default=None, help="Model yaml path.")
    parser.add_argument("--pretrained", default="yolo26n.pt", help="Float pretrained weights path.")
    parser.add_argument(
        "--qat-weights",
        dest="qat_weights",
        default="runs/detect/qat/weights/best.pt",
        help="QAT checkpoint path.",
    )
    parser.add_argument("--out", default="./runs/qat_one2many.onnx", help="ONNX output path.")
    parser.add_argument(
        "--qat-state-out",
        dest="qat_state_out",
        default="./runs/qat_one2many.pth",
        help="Saved QAT state dict path.",
    )
    parser.add_argument("--quant-config", dest="quant_config", default="./config.json", help="Quant config path.")
    parser.add_argument("--device", default="cuda", help="Export device.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[640, 640],
        help="Input image size, e.g. --imgsz 640 640 or --imgsz 640",
    )
    parser.add_argument(
        "--end2end",
        dest="end2end",
        default=False,
        type=lambda x: x.lower() == "true",
        help="Export one2one branch (True) or one2many branch (False) for NMS.",
    )
    return parser.parse_args()


class DetectOne2OneWrapper(torch.nn.Module):
    """Export only one-to-one branch outputs for QAT detection inference (per-scale, no concat)."""

    def __init__(self, qat_model: torch.nn.Module):
        super().__init__()
        self.qat_model = qat_model

    def forward(self, x: torch.Tensor):
        preds = self.qat_model(x)
        o2o = preds.get("one2one", preds)
        if isinstance(o2o, list):
            return (
                o2o[0]["boxes"],
                o2o[0]["scores"],
                o2o[1]["boxes"],
                o2o[1]["scores"],
                o2o[2]["boxes"],
                o2o[2]["scores"],
            )
        return o2o["boxes"], o2o["scores"]


class DetectOne2ManyWrapper(torch.nn.Module):
    """Export one-to-many branch outputs for QAT detection inference (per-scale, no concat, no feats)."""

    def __init__(self, qat_model: torch.nn.Module):
        super().__init__()
        self.qat_model = qat_model

    def forward(self, x: torch.Tensor):
        preds = self.qat_model(x)
        o2m = preds.get("one2many", preds)
        if isinstance(o2m, list):
            return (
                o2m[0]["boxes"],
                o2m[0]["scores"],
                o2m[1]["boxes"],
                o2m[1]["scores"],
                o2m[2]["boxes"],
                o2m[2]["scores"],
            )
        return o2m["boxes"], o2m["scores"]


class SegmentOne2OneWrapper(torch.nn.Module):
    """Export one-to-one branch outputs and proto tensors for QAT segmentation inference."""

    def __init__(self, qat_model: torch.nn.Module):
        super().__init__()
        self.qat_model = qat_model

    def forward(self, x: torch.Tensor):
        preds = self.qat_model(x)["one2one"]
        proto = preds["proto"]
        proto_outputs = tuple(proto) if isinstance(proto, (tuple, list)) else (proto,)
        return (
            preds["boxes"],
            preds["scores"],
            preds["mask_coefficient"],
            preds["feats"][0],
            preds["feats"][1],
            preds["feats"][2],
            *proto_outputs,
        )


def resolve_paths(args: argparse.Namespace) -> ExportDefaults:
    defaults = DEFAULT_EXPORTS[args.task]
    return ExportDefaults(
        task=args.task,
        model=args.model or defaults.model,
        pretrained=args.pretrained or defaults.pretrained,
        qat_weights=args.qat_weights or defaults.qat_weights,
        out=args.out or defaults.out,
        qat_state_out=args.qat_state_out or defaults.qat_state_out,
    )


def normalize_imgsz(imgsz: list[int]) -> list[int]:
    if len(imgsz) == 1:
        h = w = int(imgsz[0])
    elif len(imgsz) == 2:
        h, w = (int(v) for v in imgsz)
    else:
        raise ValueError(f"Unsupported imgsz {imgsz}. Use one or two integers.")
    return [1, 3, h, w]


def remove_invalid_graph_outputs(model: onnx.ModelProto) -> list[str]:
    """Remove graph outputs that are no longer produced after slim passes."""
    producer_outputs = {output for node in model.graph.node for output in node.output if output}
    input_names = {value.name for value in model.graph.input}
    initializer_names = {value.name for value in model.graph.initializer}
    valid_output_names = producer_outputs | input_names | initializer_names

    removed = [output.name for output in model.graph.output if output.name not in valid_output_names]
    if removed:
        kept_outputs = [output for output in model.graph.output if output.name in valid_output_names]
        del model.graph.output[:]
        model.graph.output.extend(kept_outputs)
    return removed


def ensure_main_opset(proto, version: int) -> None:
    """Bump the main ONNX opset for model and local functions when exporter emits int16 Q/DQ."""
    found = False
    for opset in proto.opset_import:
        if opset.domain == "":
            opset.version = max(opset.version, version)
            found = True
    if not found:
        proto.opset_import.add(domain="", version=version)

    functions = getattr(proto, "functions", None)
    if functions:
        for function in functions:
            ensure_main_opset(function, version)


def build_quantized_model(cfg: ExportDefaults, quant_config: str, device: str, qat_onnx_imgsz: list[int]):
    model = YOLO(cfg.model, task=cfg.task).load(cfg.pretrained)

    # Load checkpoint metadata to detect training config
    if cfg.qat_weights and Path(cfg.qat_weights).exists():
        weight_dict = torch.load(cfg.qat_weights, weights_only=False)
        train_args = weight_dict.get("train_args", {})
        ckpt_quant_config = train_args.get("qat_config", None) if isinstance(train_args, dict) else None
        if ckpt_quant_config and Path(ckpt_quant_config).exists():
            quant_config = ckpt_quant_config
            print(f"Using quant config from checkpoint: {quant_config}")

    global_config, regional_configs = ax_load_config(quant_config)
    quantizer = AXQuantizer()
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)

    float_model = model.model.to(device)
    inputs = torch.rand(*qat_onnx_imgsz, device=device)
    print("start export!")
    exported_model = torch.export.export_for_training(
        float_model,
        (inputs,),
        dynamic_shapes={"x": {0: 1, 2: Dim.AUTO, 3: Dim.AUTO}},
    )
    print("export training model done!")
    exported_module = exported_model.module()

    prepared_model = prepare_qat_pt2e(exported_module, quantizer)
    print("prepared model done!")
    torch.ao.quantization.allow_exported_model_train_eval(prepared_model)

    if cfg.qat_weights and Path(cfg.qat_weights).exists():
        weight_dict = torch.load(cfg.qat_weights, weights_only=False)
        qat_weight_dict = weight_dict.get("qat_ema") or weight_dict.get("qat_model")
        if qat_weight_dict is None:
            raise KeyError("Checkpoint missing 'qat_ema' and 'qat_model' keys")
        torch.save(qat_weight_dict, cfg.qat_state_out)
        prepared_model.load_state_dict(qat_weight_dict)
        print("load_state_dict done!")
    else:
        print("skip weight loading (no checkpoint)")
    prepared_model.eval()

    quantized_model = convert_pt2e(prepared_model)
    print("convert_pt2e done!")
    return quantized_model, inputs


def get_segment_proto_names(proto) -> list[str]:
    if isinstance(proto, (tuple, list)):
        if len(proto) == 2:
            return ["proto_masks", "proto_semseg"]
        return [f"proto_{i}" for i in range(len(proto))]
    return ["proto_masks"]


def build_export_plan(task: str, quantized_model: torch.nn.Module, inputs: torch.Tensor, end2end: bool = True):
    sample_outputs = quantized_model(inputs)
    if task == "detect":
        if end2end:
            return DetectOne2OneWrapper(quantized_model), [
                "boxes_p3",
                "scores_p3",
                "boxes_p4",
                "scores_p4",
                "boxes_p5",
                "scores_p5",
            ]
        else:
            return DetectOne2ManyWrapper(quantized_model), [
                "boxes_p3",
                "scores_p3",
                "boxes_p4",
                "scores_p4",
                "boxes_p5",
                "scores_p5",
            ]
    if task == "segment":
        proto_names = get_segment_proto_names(sample_outputs["one2one"]["proto"])
        output_names = [
            "boxes",
            "scores",
            "mask_coefficient",
            "feat_p3",
            "feat_p4",
            "feat_p5",
            *proto_names,
        ]
        return SegmentOne2OneWrapper(quantized_model), output_names
    raise ValueError(f"Unsupported task {task}")


def _fix_qdq_qdq_mismatch(model: onnx.ModelProto, threshold: float = 1.02) -> int:
    """Insert Identity (requant) nodes between adjacent Q-DQ pairs with mismatched quantization.

    In a Q→DQ→Q chain, if the first Q's scale and the second Q's scale differ beyond ``threshold`` (or their zero_points
    differ), deployment tools cannot fuse them into a single requant operation. Inserting an explicit Identity node
    severs the adjacency and marks the position where a **requant** (dequantize→requantize) is required.
    """
    import numpy as np
    from onnx import helper

    init_dict = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    insertions: list[tuple] = []

    for dq_node in model.graph.node:
        if dq_node.op_type != "DequantizeLinear":
            continue
        q_first = next(
            (n for n in model.graph.node if n.op_type == "QuantizeLinear" and n.output[0] == dq_node.input[0]), None
        )
        q_second = next(
            (n for n in model.graph.node if n.op_type == "QuantizeLinear" and dq_node.output[0] in n.input), None
        )
        if q_first is None or q_second is None:
            continue

        s1 = float(init_dict.get(q_first.input[1], np.array([0])).flatten()[0])
        s2 = float(init_dict.get(q_second.input[1], np.array([0])).flatten()[0])
        if s1 < 1e-10 or s2 < 1e-10:
            continue

        zp1 = int(init_dict.get(q_first.input[2], np.array([0])).flatten()[0])
        zp2 = int(init_dict.get(q_second.input[2], np.array([0])).flatten()[0])
        scale_ratio = max(s1, s2) / min(s1, s2)
        zp_mismatch = zp1 != zp2

        if scale_ratio < threshold and not zp_mismatch:
            continue

        name = dq_node.name.replace("/", "_").replace(".", "_")
        identity_output = f"{name}_requant_id"
        identity_node = helper.make_node(
            "Identity",
            inputs=[dq_node.output[0]],
            outputs=[identity_output],
            name=f"{name}_requant",
        )
        reason = f"scale_ratio={scale_ratio:.2f}"
        if zp_mismatch:
            reason += f", zp_diff={zp1}!={zp2}"
        print(f"  requant @ {dq_node.name}: {reason}  (s1={s1:.6f}, s2={s2:.6f})")
        insertions.append((identity_node, q_second, dq_node.output[0], identity_output))

    if not insertions:
        return 0

    nodes = list(model.graph.node)
    for id_node, q_second, old_out, new_out in insertions:
        q2_idx = nodes.index(q_second)
        nodes.insert(q2_idx, id_node)
        for i, inp in enumerate(q_second.input):
            if inp == old_out:
                q_second.input[i] = new_out
                break

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return len(insertions)


def _merge_adjacent_dq_q(model: onnx.ModelProto, threshold: float = 1.02) -> int:
    """Merge adjacent DQ→Q pairs with nearly identical scales into a single pass-through.

    PT2E may insert redundant DQ→Q pairs between adjacent layers when their observer scales differ slightly (< 2%).
    These pairs are a no-op passthrough that can be eliminated.
    """
    import numpy as np

    init_dict = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    nodes_to_remove = set()
    merged = 0

    for dq_node in list(model.graph.node):
        if dq_node.op_type != "DequantizeLinear":
            continue

        # Find Q that directly consumes this DQ output
        q_next = next(
            (n for n in model.graph.node if n.op_type == "QuantizeLinear" and dq_node.output[0] == n.input[0]), None
        )
        if q_next is None:
            continue

        # Ensure DQ has exactly one consumer (the Q_next)
        dq_consumers = [n for n in model.graph.node if dq_node.output[0] in n.input]
        if len(dq_consumers) != 1:
            continue

        # Find Q that feeds this DQ
        q_prev = next(
            (n for n in model.graph.node if n.op_type == "QuantizeLinear" and n.output[0] == dq_node.input[0]), None
        )
        if q_prev is None:
            continue

        # Find DQ that Q_next feeds, ensure Q_next has single consumer
        q_next_consumers = [n for n in model.graph.node if q_next.output[0] in n.input]
        if len(q_next_consumers) != 1:
            continue
        dq_next = q_next_consumers[0]
        if dq_next.op_type != "DequantizeLinear":
            continue

        # Check scale proximity
        s_dq = float(init_dict.get(dq_node.input[1], np.array([0])).flatten()[0])
        s_q = float(init_dict.get(q_next.input[1], np.array([0])).flatten()[0])
        if s_dq < 1e-10 or s_q < 1e-10:
            continue
        if max(s_dq, s_q) / min(s_dq, s_q) >= threshold:
            continue

        # Check zero_point consistency (affine quantization may have non-zero zp)
        zp_dq = int(init_dict.get(dq_node.input[2], np.array([0])).flatten()[0])
        zp_q = int(init_dict.get(q_next.input[2], np.array([0])).flatten()[0])
        if zp_dq != zp_q:
            continue

        # Merge: skip DQ+Q_next, rewire Q_prev output to DQ_next input
        dq_next.input[0] = q_prev.output[0]
        nodes_to_remove.add(dq_node.name)
        nodes_to_remove.add(q_next.name)
        merged += 1

    if nodes_to_remove:
        new_nodes = [n for n in model.graph.node if n.name not in nodes_to_remove]
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    # Second pass: handle DQ with multiple consumers where one consumer is a redundant Q->DQ branch
    for dq_node in list(model.graph.node):
        if dq_node.op_type != "DequantizeLinear":
            continue

        dq_consumers = [n for n in model.graph.node if dq_node.output[0] in n.input]
        s_dq = float(init_dict.get(dq_node.input[1], np.array([0])).flatten()[0])
        if s_dq < 1e-10:
            continue

        for consumer in dq_consumers:
            if consumer.op_type != "QuantizeLinear":
                continue

            s_q = float(init_dict.get(consumer.input[1], np.array([0])).flatten()[0])
            if s_q < 1e-10:
                continue
            if max(s_dq, s_q) / min(s_dq, s_q) >= threshold:
                continue

            # Check zero_point consistency
            zp_dq = int(init_dict.get(dq_node.input[2], np.array([0])).flatten()[0])
            zp_q = int(init_dict.get(consumer.input[2], np.array([0])).flatten()[0])
            if zp_dq != zp_q:
                continue

            # Find the DQ that this intermediate Q feeds
            q_consumers = [n for n in model.graph.node if consumer.output[0] in n.input]
            if len(q_consumers) != 1:
                continue
            dq_mid = q_consumers[0]
            if dq_mid.op_type != "DequantizeLinear":
                continue

            # Ensure dq_mid has at least one consumer that would benefit from bypass
            dq_mid_consumers = [n for n in model.graph.node if dq_mid.output[0] in n.input]

            # Rewire dq_mid consumers to use parent DQ output directly
            for ds_consumer in dq_mid_consumers:
                for i, inp in enumerate(ds_consumer.input):
                    if inp == dq_mid.output[0]:
                        ds_consumer.input[i] = dq_node.output[0]
                        break

            nodes_to_remove.add(consumer.name)
            nodes_to_remove.add(dq_mid.name)
            merged += 1

    if nodes_to_remove:
        new_nodes = [n for n in model.graph.node if n.name not in nodes_to_remove]
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    return merged


def _split_shared_mul_dequant(model: onnx.ModelProto) -> int:
    """Split DequantizeLinear nodes that feed into multiple Mul ops.

    When a single DQ output is consumed by two or more Mul nodes, deployment tools may fail to relate the separate
    DQ→Mul→Q chains back to a shared quantized tensor. Creating per-Mul copies of the DQ resolves this.
    """
    from onnx import helper

    {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    new_nodes = []
    split_count = 0

    for dq_node in model.graph.node:
        if dq_node.op_type != "DequantizeLinear":
            continue
        mul_consumers = [n for n in model.graph.node if n.op_type == "Mul" and dq_node.output[0] in n.input]
        if len(mul_consumers) <= 1:
            continue

        dq_out = dq_node.output[0]
        scale_name = dq_node.input[1]
        zp_name = dq_node.input[2] if len(dq_node.input) > 2 else None

        for mul_node in mul_consumers[1:]:
            new_output = f"{dq_node.output[0]}_split_{mul_node.name.split('/')[-1]}"
            new_dq = helper.make_node(
                "DequantizeLinear",
                inputs=[dq_node.input[0], scale_name] + ([zp_name] if zp_name else []),
                outputs=[new_output],
                name=f"{dq_node.name}_split_{split_count}",
            )
            new_nodes.append((new_dq, mul_node))
            for i, inp in enumerate(mul_node.input):
                if inp == dq_out:
                    mul_node.input[i] = new_output
                    break
            split_count += 1

    if new_nodes:
        nodes = list(model.graph.node)
        for new_dq, mul_node in new_nodes:
            idx = nodes.index(mul_node)
            nodes.insert(idx, new_dq)  # insert before its Mul consumer
        del model.graph.node[:]
        model.graph.node.extend(nodes)

    return split_count


def export_onnx(export_wrapper: torch.nn.Module, inputs: torch.Tensor, out_path: str, output_names: list[str]) -> None:
    onnx_program = torch.onnx.export(export_wrapper, (inputs,), dynamo=True, opset_version=21)
    onnx_program.optimize()
    onnx_program.save(out_path)
    print(f"export qat model to [{out_path}] done!")

    model_simp = slim(onnx.load(out_path))
    ensure_main_opset(model_simp, REQUIRED_ONNX_OPSET)
    removed_outputs = remove_invalid_graph_outputs(model_simp)
    n_fixed = _fix_qdq_qdq_mismatch(model_simp)
    if n_fixed:
        print(f"inserted {n_fixed} requant marker(s) where Q→DQ→Q scales diverge")
    n_split = _split_shared_mul_dequant(model_simp)
    if n_split:
        print(f"split {n_split} shared DequantizeLinear(s) for independent Mul ops")
    n_merged = _merge_adjacent_dq_q(model_simp)
    if n_merged:
        print(f"merged {n_merged} redundant DQ→Q pair(s) with near-identical scales")
    sim_path = out_path.replace(".onnx", "_slim.onnx")
    onnx.save(model_simp, sim_path)
    onnx.checker.check_model(model_simp)
    if removed_outputs:
        print(f"remove invalid slim outputs: {removed_outputs}")
    print(f"save onnx model to [{sim_path}] Successfully!")


def main() -> None:
    args = parse_args()
    cfg = resolve_paths(args)
    qat_onnx_imgsz = normalize_imgsz(args.imgsz)
    output_path = str(Path(cfg.out))

    print(
        f"task={cfg.task}, model={cfg.model}, pretrained={cfg.pretrained}, "
        f"qat_weights={cfg.qat_weights}, out={output_path}"
    )

    quantized_model, inputs = build_quantized_model(cfg, args.quant_config, args.device, qat_onnx_imgsz)
    export_wrapper, output_names = build_export_plan(cfg.task, quantized_model, inputs, end2end=args.end2end)
    export_onnx(export_wrapper, inputs, output_path, output_names)


if __name__ == "__main__":
    main()
