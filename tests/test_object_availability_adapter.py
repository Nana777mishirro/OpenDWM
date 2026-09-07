import unittest

import torch

from dwm.models.object_availability_adapter import ObjectAvailabilityAdapter


class ObjectAvailabilityAdapterTest(unittest.TestCase):
    def test_constant_velocity_annotation_uses_observed_displacement(self):
        from dwm.datasets.availability import ObjectAvailabilityDataset

        previous = {
            "translation": [1.0, 2.0, 3.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "size": [4.0, 5.0, 6.0],
        }
        last = {
            "translation": [3.0, 1.0, 3.5],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "size": [4.0, 5.0, 6.0],
        }
        predicted = ObjectAvailabilityDataset._constant_velocity_annotation(
            previous, last, frames_after_last=3)

        self.assertEqual(predicted["translation"], [9.0, -2.0, 5.0])
        self.assertEqual(predicted["rotation"], last["rotation"])
        self.assertEqual(predicted["size"], last["size"])
        self.assertEqual(last["translation"], [3.0, 1.0, 3.5])

    def test_constant_acceleration_annotation_uses_three_observations(self):
        from dwm.datasets.availability import ObjectAvailabilityDataset

        older = {
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "size": [4.0, 5.0, 6.0],
        }
        previous = {**older, "translation": [1.0, 0.0, 0.0]}
        last = {**older, "translation": [3.0, 0.0, 0.0]}
        predicted = \
            ObjectAvailabilityDataset._constant_acceleration_annotation(
                older, previous, last, frames_after_last=2,
                acceleration_scale=1.0)

        self.assertEqual(predicted["translation"], [10.0, 0.0, 0.0])
        self.assertEqual(last["translation"], [3.0, 0.0, 0.0])

    def test_rotation_extrapolation_repeats_observed_turn(self):
        import numpy as np
        import transforms3d
        from dwm.datasets.availability import ObjectAvailabilityDataset

        previous = {"rotation": [1.0, 0.0, 0.0, 0.0]}
        last = {"rotation": transforms3d.euler.euler2quat(
            0, 0, np.pi / 2).tolist()}
        predicted = {"rotation": list(last["rotation"])}
        ObjectAvailabilityDataset._extrapolate_annotation_rotation(
            previous, last, predicted, frames_after_last=1)
        predicted_matrix = transforms3d.quaternions.quat2mat(
            predicted["rotation"])
        expected_matrix = transforms3d.euler.euler2mat(0, 0, np.pi)

        self.assertTrue(np.allclose(
            predicted_matrix, expected_matrix, atol=1e-7))

    def test_explicit_absence_zeroes_projection_bias_residual(self):
        batch_size, time, views, spatial, hidden = 1, 4, 2, 4, 8
        adapter = ObjectAvailabilityAdapter(
            hidden_dim=hidden,
            num_classes=3,
            class_embedding_dim=4,
            slot_dim=6,
            injection_layers=(0,),
            validate_inputs=True,
        )
        with torch.no_grad():
            adapter.residual_projections["0"].weight.fill_(0.25)
            adapter.residual_projections["0"].bias.fill_(0.5)

        hidden_states = torch.randn(
            batch_size * time * views, spatial, hidden)
        class_ids = torch.ones(batch_size, time, 1, dtype=torch.long)
        box_states = torch.randn(batch_size, time, 1, 10)
        availability = torch.tensor([[[True], [False], [False], [True]]])
        slot_mask = torch.tensor([[[True], [True], [False], [True]]])
        source_frames = torch.tensor([[[0], [0], [-1], [3]]])
        spatial_prior = torch.zeros(batch_size, time, views, 2, 2)
        spatial_prior[:, 1, :, 0, 0] = 1

        residual = adapter(
            hidden_states,
            batch_size=batch_size,
            sequence_length=time,
            view_count=views,
            class_ids=class_ids,
            box_states=box_states,
            availability=availability,
            slot_mask=slot_mask,
            layer_index=0,
            source_frame_indices=source_frames,
            spatial_prior=spatial_prior,
            spatial_height=2,
            spatial_width=2,
        ).view(batch_size, time, views, spatial, hidden)

        self.assertEqual(torch.count_nonzero(residual[:, 0]).item(), 0)
        self.assertGreater(torch.count_nonzero(residual[:, 1, :, 0]).item(), 0)
        self.assertEqual(torch.count_nonzero(residual[:, 1, :, 1:]).item(), 0)
        self.assertEqual(torch.count_nonzero(residual[:, 2]).item(), 0)
        self.assertEqual(torch.count_nonzero(residual[:, 3]).item(), 0)

    def test_spatial_prior_token_dilation_expands_gate(self):
        batch_size, time, views, spatial, hidden = 1, 2, 1, 9, 8
        adapter = ObjectAvailabilityAdapter(
            hidden_dim=hidden,
            num_classes=3,
            class_embedding_dim=4,
            slot_dim=6,
            injection_layers=(0,),
            spatial_prior_dilation_tokens=1,
        )
        with torch.no_grad():
            adapter.residual_projections["0"].weight.fill_(0.25)
            adapter.residual_projections["0"].bias.fill_(0.5)
        residual = adapter(
            torch.randn(batch_size * time * views, spatial, hidden),
            batch_size=batch_size,
            sequence_length=time,
            view_count=views,
            class_ids=torch.ones(batch_size, time, 1, dtype=torch.long),
            box_states=torch.randn(batch_size, time, 1, 10),
            availability=torch.tensor([[[True], [False]]]),
            slot_mask=torch.ones(batch_size, time, 1, dtype=torch.bool),
            layer_index=0,
            source_frame_indices=torch.tensor([[[0], [0]]]),
            spatial_prior=torch.tensor([[[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                                         [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]]),
            spatial_height=3,
            spatial_width=3,
        ).view(batch_size, time, views, spatial, hidden)

        self.assertEqual(torch.count_nonzero(residual[:, 0]).item(), 0)
        self.assertEqual(
            torch.count_nonzero(residual[:, 1]).item(), spatial * hidden)


if __name__ == "__main__":
    unittest.main()
