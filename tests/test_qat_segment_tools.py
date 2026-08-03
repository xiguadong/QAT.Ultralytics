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


class _OBBOutputs(torch.nn.Module):
    def forward(self, x):
        return {
            "one2one": {
                "boxes": x[:, :2],
                "scores": x[:, :1],
                "angle": x[:, :1],
                "feats": [x, x, x],
            }
        }


class _PoseOutputs(torch.nn.Module):
    def forward(self, x):
        return {
            "one2one": {
                "boxes": x[:, :2],
                "scores": x[:, :1],
                "kpts": x[:, :3],
                "kpts_sigma": x[:, :2],
                "feats": [x, x, x],
            }
        }


class _ClassifyOutputs(torch.nn.Module):
    def forward(self, x):
        # The train-traced Classify graph outputs a single (batch, nc) logits tensor.
        return x.flatten(1)[:, :5]


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


def test_obb_export_plan_preserves_loss_compatible_concatenated_outputs():
    model = _OBBOutputs()
    inputs = torch.rand(1, 3, 8, 8)

    wrapper, names = export_module.build_export_plan("obb", model, inputs)
    outputs = wrapper(inputs)

    assert names == ["boxes", "scores", "angle"]
    assert len(outputs) == len(names)
    assert outputs[0].shape[1] == 2


def test_pose_export_plan_preserves_deploy_keypoint_outputs():
    model = _PoseOutputs()
    inputs = torch.rand(1, 3, 8, 8)

    wrapper, names = export_module.build_export_plan("pose", model, inputs)
    outputs = wrapper(inputs)

    assert names == ["boxes", "scores", "keypoints"]
    assert len(outputs) == len(names)
    assert outputs[2].shape[1] == 3


def test_classify_export_plan_emits_single_logits_output():
    model = _ClassifyOutputs()
    inputs = torch.rand(1, 3, 8, 8)

    wrapper, names = export_module.build_export_plan("classify", model, inputs)
    outputs = wrapper(inputs)

    # Deploy contract: raw logits (no softmax tail), fully quantized single output.
    assert names == ["logits"]
    assert outputs.ndim == 2 and outputs.shape[1] == 5
    assert torch.equal(outputs, model(inputs))
