import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

from dwm.pipelines.ctsd import CrossviewTemporalSD
from dwm.tools.object_availability_paired_validation import \
    resolve_checkpoint_kind


class ObjectAvailabilityCheckpointTest(unittest.TestCase):
    @staticmethod
    def make_model():
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(3, 3)
        model.object_availability_adapter = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 3),
        )
        return model

    def test_extracts_only_adapter_tensors(self):
        model = self.make_model()
        adapter_state = \
            CrossviewTemporalSD.get_object_availability_adapter_state(
                model.state_dict())

        self.assertTrue(adapter_state)
        self.assertTrue(all(
            key.startswith("object_availability_adapter.")
            for key in adapter_state))
        self.assertFalse(any(
            key.startswith("backbone.") for key in adapter_state))

    def test_adapter_overlay_does_not_change_backbone(self):
        source = self.make_model()
        target = self.make_model()
        backbone_before = {
            key: value.clone()
            for key, value in target.backbone.state_dict().items()
        }
        with torch.no_grad():
            for parameter in source.object_availability_adapter.parameters():
                parameter.fill_(0.25)

        adapter_state = \
            CrossviewTemporalSD.get_object_availability_adapter_state(
                source.state_dict())
        CrossviewTemporalSD.load_object_availability_adapter_state(
            target, adapter_state, "test.pth")

        for key, value in target.backbone.state_dict().items():
            self.assertTrue(torch.equal(value, backbone_before[key]))
        for value in target.object_availability_adapter.state_dict().values():
            self.assertTrue(torch.equal(value, torch.full_like(value, 0.25)))

    def test_rejects_incomplete_adapter_checkpoint(self):
        model = self.make_model()
        adapter_state = \
            CrossviewTemporalSD.get_object_availability_adapter_state(
                model.state_dict())
        adapter_state.pop(next(iter(adapter_state)))

        with self.assertRaisesRegex(ValueError, "missing keys"):
            CrossviewTemporalSD.load_object_availability_adapter_state(
                model, adapter_state, "incomplete.pth")

    def test_save_checkpoint_writes_only_adapter_tensors(self):
        pipeline = CrossviewTemporalSD.__new__(CrossviewTemporalSD)
        pipeline.training_config = {
            "save_object_availability_adapter_only": True,
        }
        pipeline.distribution_framework = "ddp"
        pipeline.should_save = True
        pipeline.model = self.make_model()
        pipeline.model_wrapper = pipeline.model
        pipeline.optimizer = None

        with tempfile.TemporaryDirectory() as output_path, mock.patch(
            "dwm.distributed.distributed_save_optimizer_state"
        ) as save_optimizer:
            pipeline.save_checkpoint(output_path, 7)
            state_dict = torch.load(
                f"{output_path}/checkpoints/7.pth",
                map_location="cpu", weights_only=True)

        self.assertTrue(state_dict)
        self.assertTrue(all(
            key.startswith("object_availability_adapter.")
            for key in state_dict))
        save_optimizer.assert_called_once()

    def test_paired_validation_detects_adapter_only_checkpoint(self):
        model = self.make_model()
        adapter_state = \
            CrossviewTemporalSD.get_object_availability_adapter_state(
                model.state_dict())
        with tempfile.TemporaryDirectory() as output_path:
            checkpoint_path = f"{output_path}/adapter.pth"
            torch.save(adapter_state, checkpoint_path)
            kind = resolve_checkpoint_kind(
                Path(checkpoint_path), "auto")

        self.assertEqual(kind, "adapter")


if __name__ == "__main__":
    unittest.main()
