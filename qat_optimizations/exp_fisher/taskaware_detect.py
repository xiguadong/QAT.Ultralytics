"""T1: 检测头输出激活的"任务加权"IFMR 校准。.

3 个检测头 conv 的输出激活是 per-tensor 量化(ACTIVATED)、覆盖 255=3×85 通道，
被占 ~95% 误差的类别通道主导。本模块对这 3 个输出 scale 用**通道加权 MSE** 重新搜索：
  Cost(s) = Σ_c w_c · mean_n( (qdq(out_c, s) - out_c)^2 )
通道权重按 c%85 分组：box(0-3)=w_box, obj(4)=w_obj, cls(5-84)=w_cls。
压低 w_cls → scale 偏向保住 box/obj（牺牲类别幅度）。非破坏性，只改校准 scale。
"""

import torch
from quant.ppq.core import QuantizationStates
from quant.ppq.executor import TorchExecutor
from quant.ppq.IR import QuantableOperation


def find_detect_convs(graph):
    """从图输出回溯最近的 Conv（255 通道检测头）。."""

    def src(var):
        return getattr(var, "source_op", None)

    def back(op, d=0):
        if op is None or d > 6:
            return None
        if op.type == "Conv":
            return op
        for inp in op.inputs:
            if not inp.is_parameter:
                r = back(src(inp), d + 1)
                if r:
                    return r
        return None

    convs, seen = [], set()
    for var in graph.outputs.values():
        c = back(src(var))
        if c is not None and c.name not in seen:
            convs.append(c)
            seen.add(c.name)
    return convs


def set_detect_weights_fp32(quant_graph, float_graph):
    """诊断: 把 3 个检测头 conv 的权重换回浮点(从 float_graph 取)并设 FP32, 隔离权重量化损失。."""
    from quant.ppq.core import QuantizationStates

    fmap = {c.name: c for c in find_detect_convs(float_graph)}
    n = 0
    for c in find_detect_convs(quant_graph):
        if c.name in fmap:
            c.inputs[1].value = fmap[c.name].inputs[1].value.clone()
            c.config.input_quantization_config[1].state = QuantizationStates.FP32
            n += 1
    return n


def _sigmoid_deriv(z):
    s = torch.sigmoid(z)
    return s * (1.0 - s)


@torch.no_grad()
def fisher_detect_refine(
    graph,
    dataloader,
    collate_fn,
    device="cpu",
    no=85,
    gate_thresh=0.05,
    s_lo=0.25,
    s_hi=1.8,
    s_step=0.01,
    w_box=1.0,
    w_obj=1.0,
    w_cls=1.0,
    max_batches=64,
    verbose=False,
):
    """推导出的任务感知 cost: 对 3 个检测头输出 scale， 搜 s 最小化 Σ_i w_i·(x_i - Q_s(x_i))²，其中 w_i = gate_a · σ'(x_{f,i})² · r_component。
    gate_a = σ(obj_f) (前景门控, 背景自动≈0) σ'(z)=σ(z)(1-σ(z)) (量化在 logit, 任务在概率 → 概率敏感区加权) r: box/obj/argmax-cls 有权,
    非argmax的79类自动=0 只在前景 anchor-location 上收集元素（背景权重≈0, 直接丢弃以省内存并聚焦）。.
    """
    convs = find_detect_convs(graph)
    names = [c.outputs[0].name for c in convs]
    gf = graph.copy()
    for op in gf.operations.values():
        if isinstance(op, QuantableOperation):
            op.dequantize(activation_only=True)
    fexec = TorchExecutor(gf, device=device)
    next(iter(gf.inputs))
    store = {n: {"x": [], "w": []} for n in names}

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        b = collate_fn(batch) if collate_fn is not None else batch
        outs = fexec.forward(b, output_names=names)
        for n, o in zip(names, outs):  # o: [1,255,H,W]
            C = o.shape[1]
            A = C // no
            oo = o.reshape(A, no, o.shape[2], o.shape[3])  # [A,85,H,W] logits
            gate = torch.sigmoid(oo[:, 4])  # [A,H,W] objectness 概率
            fg = gate > gate_thresh
            if fg.sum() == 0:
                continue
            sigp2 = _sigmoid_deriv(oo) ** 2  # [A,85,H,W]
            r = torch.zeros_like(oo)
            r[:, 0:4] = w_box
            r[:, 4] = w_obj
            amax = oo[:, 5:].argmax(dim=1, keepdim=True)  # [A,1,H,W]
            onehot = torch.zeros_like(oo[:, 5:]).scatter_(1, amax, 1.0)
            r[:, 5:] = w_cls * onehot
            w = gate.unsqueeze(1) * sigp2 * r  # [A,85,H,W]
            fgm = fg.unsqueeze(1).expand_as(w)
            store[n]["x"].append(oo[fgm].flatten().cpu())
            store[n]["w"].append(w[fgm].flatten().cpu())

    ratios = []
    rr = s_lo
    while rr < s_hi:
        ratios.append(rr)
        rr += s_step

    for conv in convs:
        oc = conv.config.output_quantization_config[0]
        if oc.scale is None or not QuantizationStates.is_activated(oc.state):
            continue
        xs = store[conv.outputs[0].name]["x"]
        if not xs:
            continue
        x = torch.cat(xs)
        w = torch.cat(store[conv.outputs[0].name]["w"])
        qmin, qmax = oc.quant_min, oc.quant_max
        off = float(oc.offset.item()) if (oc.offset is not None and oc.offset.numel() == 1) else 0.0
        base = float(oc.scale.item())
        best = (float("inf"), base)
        for ratio in ratios:
            s = max(base * ratio, 1e-12)
            q = torch.clamp(torch.round(x / s) + off, qmin, qmax)
            qdq = (q - off) * s
            cost = float((w * (qdq - x) ** 2).sum())
            if cost < best[0]:
                best = (cost, s)
        oc.scale = torch.tensor(best[1], dtype=torch.float32, device=device)
        if verbose:
            print(f"[Fisher] {conv.name}: base={base:.5f} -> {best[1]:.5f} (前景元素={x.numel()})")
    return graph


@torch.no_grad()
def taskaware_detect_refine(
    graph,
    dataloader,
    collate_fn,
    device="cpu",
    w_box=1.0,
    w_obj=1.0,
    w_cls=1.0,
    no=85,
    s_lo=0.3,
    s_hi=1.6,
    s_step=0.02,
    max_batches=64,
    sub=2048,
    verbose=False,
):
    convs = find_detect_convs(graph)
    names = [c.outputs[0].name for c in convs]

    # 浮点检测头输出（独立浮点图，按通道保留，spatial 子采样）
    [op for op in graph.operations.values() if isinstance(op, QuantableOperation)]
    gf = graph.copy()
    for op in gf.operations.values():
        if isinstance(op, QuantableOperation):
            op.dequantize(activation_only=True)
    fexec = TorchExecutor(gf, device=device)
    next(iter(gf.inputs))
    cache = {n: [] for n in names}
    g = torch.Generator().manual_seed(123)
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        b = collate_fn(batch) if collate_fn is not None else batch
        outs = fexec.forward(b, output_names=names)
        for n, o in zip(names, outs):  # o: [1, 255, H, W]
            o2 = o.reshape(o.shape[1], -1)  # [255, H*W]
            idx = torch.randint(0, o2.shape[1], (min(sub, o2.shape[1]),), generator=g)
            cache[n].append(o2[:, idx].cpu())
    cache = {n: torch.cat(v, dim=1) for n, v in cache.items()}  # [255, total]

    ratios = []
    r = s_lo
    while r < s_hi:
        ratios.append(r)
        r += s_step

    for conv in convs:
        out_cfg = conv.config.output_quantization_config[0]
        if out_cfg.scale is None or not QuantizationStates.is_activated(out_cfg.state):
            if verbose:
                print(f"[T1] 跳过 {conv.name}（输出 cfg 不可改）")
            continue
        data = cache[conv.outputs[0].name]  # [255, total]
        C = data.shape[0]
        # 通道权重向量 by c%no
        wvec = torch.ones(C)
        for c in range(C):
            comp = c % no
            wvec[c] = w_box if comp < 4 else (w_obj if comp == 4 else w_cls)
        wvec = wvec / wvec.sum()

        qmin, qmax = out_cfg.quant_min, out_cfg.quant_max
        offset = float(out_cfg.offset.item()) if out_cfg.offset is not None and out_cfg.offset.numel() == 1 else 0.0
        base = float(out_cfg.scale.item())
        best = (float("inf"), base)
        for ratio in ratios:
            s = max(base * ratio, 1e-12)
            q = torch.clamp(torch.round(data / s) + offset, qmin, qmax)
            qdq = (q - offset) * s
            mse_c = ((qdq - data) ** 2).mean(dim=1)  # [255]
            cost = float((wvec * mse_c).sum())
            if cost < best[0]:
                best = (cost, s)
        out_cfg.scale = torch.tensor(best[1], dtype=torch.float32, device=device)
        if verbose:
            print(f"[T1] {conv.name}: base_scale={base:.5f} -> {best[1]:.5f} (w_cls={w_cls})")
    return graph
