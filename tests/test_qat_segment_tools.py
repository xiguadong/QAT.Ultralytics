import torch

import export as export_module


class _SegmentOutputs(torch.nn.Module):
    def forward(self, x):
        return {
            "one2one": {
                "boxes": [x[:, :1], x[:, :1], x[:, :1]],
                "scores": [x[:, :1], x[:, :1], x[:, :1]],
                "mask_coefficient": x[:, :1],
                "feats": [x, x, x],
                "proto": (x[:, :1], x[:, :2]),
            }
        }


def test_segment_export_plan_flattens_outputs_without_feature_maps():
    model = _SegmentOutputs()
    inputs = torch.rand(1, 3, 8, 8)

    wrapper, names = export_module.build_export_plan("segment", model, inputs)
    outputs = wrapper(inputs)

    assert names == [
        "boxes_p3",
        "boxes_p4",
        "boxes_p5",
        "scores_p3",
        "scores_p4",
        "scores_p5",
        "mask_coefficient",
        "proto_masks",
        "proto_semseg",
    ]
    assert len(outputs) == len(names)
    assert all(output is not inputs for output in outputs)
