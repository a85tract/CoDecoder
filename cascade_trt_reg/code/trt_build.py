"""CASCADE-TRT step 1 (exploratory): export a Cascade network's evaluation computation to ONNX at one fixed batch size
and build a TensorRT plan (BF16 flag; FP16 only if the BF16 build fails; no FP8, no INT8).

    python trt_build.py --net bb --ckpt C14p.pt --batch 4096 --out DIR
    python trt_build.py --net d13 --batch 16 --out DIR          (CASCADE_ROOT as surface.sv_common reads it)

Writes DIR/<net>_b<B>.onnx, DIR/<net>_b<B>.plan, DIR/<net>_b<B>.layers.json (engine inspector) and appends one JSON row
to DIR/build.jsonl (exporter, TensorRT version, flags, build seconds, sha256 of both files, layer precision counts)."""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trt_common as tc  # noqa: E402

D_BB = 432


def export_onnx(net, model, B, Dd, path, dev):
    import torch
    mod = tc.export_module(net, model)
    x = torch.zeros(B, Dd, dtype=torch.uint8, device=dev)
    names = list(tc.OUT_NAMES[net])
    errs = []
    with torch.no_grad():
        try:
            torch.onnx.export(mod, (x,), path, input_names=["det"], output_names=names, opset_version=17,
                              do_constant_folding=True, dynamo=False)
            return "torchscript", errs
        except Exception as e:                              # noqa: BLE001
            errs.append("torchscript: " + repr(e)[:400])
        try:
            torch.onnx.export(mod, (x,), path, input_names=["det"], output_names=names, opset_version=18, dynamo=True)
            return "dynamo", errs
        except Exception as e:                              # noqa: BLE001
            errs.append("dynamo: " + repr(e)[:400])
    raise SystemExit("ONNX export failed: " + " | ".join(errs))


def build_plan(onnx_path, plan_path, prec, opt_level, ws_gb, conv_prec=False):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise SystemExit("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    n_conv = 0
    if conv_prec:                                        # the convolutions held to the build precision (no TF32 path)
        dt = trt.DataType.BF16 if prec == "bf16" else trt.DataType.HALF
        for i in range(network.num_layers):
            ly = network.get_layer(i)
            if ly.type == trt.LayerType.CONVOLUTION:
                ly.precision = dt
                ly.set_output_type(0, dt)
                n_conv += 1
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    if ws_gb > 0:                                        # 0: TensorRT's default (the device's memory)
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(ws_gb * (1 << 30)))
    cfg.set_flag(trt.BuilderFlag.BF16 if prec == "bf16" else trt.BuilderFlag.FP16)
    cfg.builder_optimization_level = opt_level
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    t0 = time.time()
    ser = builder.build_serialized_network(network, cfg)
    secs = time.time() - t0
    if ser is None:
        return None, secs
    if conv_prec:
        print(json.dumps({"conv_layers_constrained": n_conv, "prec": prec}), flush=True)
    with open(plan_path, "wb") as fh:
        fh.write(ser)
    return network.num_layers, secs


def inspect(plan_path, layers_path):
    """Engine inspector JSON -> (layer count, counts of the output tensors' formats, e.g. 'Row major linear BF16')."""
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(plan_path, "rb") as fh:
        eng = rt.deserialize_cuda_engine(fh.read())
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    with open(layers_path, "w") as fh:
        json.dump(info, fh, indent=1)
    fmts, kinds = collections.Counter(), collections.Counter()
    for ly in info.get("Layers", []):
        if isinstance(ly, dict):
            kinds[ly.get("LayerType", "?")] += 1
            for o in ly.get("Outputs", []):
                fmts[o.get("Format/Datatype", "?")] += 1
    return len(info.get("Layers", [])), dict(fmts), dict(kinds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, choices=["bb", "d13"])
    ap.add_argument("--ckpt", default="", help="bb: the Cascade checkpoint (C14p)")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opt-level", type=int, default=3, help="TensorRT builder optimisation level (default 3)")
    ap.add_argument("--conv-prec", action="store_true", help="hold every convolution layer to the build precision "
                    "(OBEY_PRECISION_CONSTRAINTS): without it TensorRT 10.16 built d = 13 in FP32/TF32")
    ap.add_argument("--ws-gb", type=float, default=0.0, help="workspace cap in GB (0: TensorRT's default)")
    a = ap.parse_args()
    import tensorrt as trt
    import torch
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    model, sha = tc.load_net(a.net, dev, a.ckpt)
    Dd = D_BB if a.net == "bb" else model.D
    stem = os.path.join(a.out, f"{a.net}_b{a.batch}")
    t0 = time.time()
    exporter, errs = export_onnx(a.net, model, a.batch, Dd, stem + ".onnx", dev)
    t_export = time.time() - t0
    del model
    torch.cuda.empty_cache()
    tried = []
    for prec in ("bf16", "fp16"):
        n_layers, secs = build_plan(stem + ".onnx", stem + ".plan", prec, a.opt_level, a.ws_gb, a.conv_prec)
        tried.append({"prec": prec, "ok": n_layers is not None, "build_s": round(secs, 1)})
        if n_layers is not None:
            break
    else:
        raise SystemExit(f"TensorRT build failed in bf16 and fp16: {tried}")
    n_eng, fmts, kinds = inspect(stem + ".plan", stem + ".layers.json")
    tc.emit(os.path.join(a.out, "build.jsonl"), {
        "net": a.net, "batch": a.batch, "ckpt_sha256": sha, "D": Dd, "exporter": exporter, "export_errors": errs,
        "export_s": round(t_export, 1), "onnx_sha256": tc.sha256_file(stem + ".onnx"),
        "onnx_bytes": os.path.getsize(stem + ".onnx"), "prec": prec, "tried": tried, "opt_level": a.opt_level, "ws_gb": a.ws_gb, "conv_prec": a.conv_prec,
        "network_layers": n_layers, "engine_layers": n_eng, "out_formats": fmts, "layer_types": kinds,
        "plan_sha256": tc.sha256_file(stem + ".plan"), "plan_bytes": os.path.getsize(stem + ".plan"),
        "trt": trt.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})


if __name__ == "__main__":
    main()
