"""Validity-aware persistent object conditioning for CTSD.

This module is deliberately independent from the existing raster adapter.  It
keeps an object slot alive while a measurement is unavailable and removes the
slot only for explicit absence.  The final projection is zero initialized so
enabling the module on an existing checkpoint is initially a no-op.
"""

from typing import Iterable, Optional

import torch


class ObjectAvailabilityAdapter(torch.nn.Module):
    """Turn persistent object observations into transformer residuals.

    Input tensor shapes are ``[batch, time, slots, ...]``.  Slot identity is
    represented by a stable slot index across time; arbitrary nuScenes token
    strings are intentionally not embedded by the model.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 12,
        box_dim: int = 10,
        class_embedding_dim: int = 32,
        slot_dim: int = 256,
        injection_layers: Iterable[int] = (0,),
        inject_on_observed: bool = False,
        use_spatial_prior: bool = True,
        spatial_prior_dilation_tokens: int = 0,
        use_velocity: bool = False,
        velocity_dim: int = 3,
        validate_inputs: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.box_dim = box_dim
        self.slot_dim = slot_dim
        self.injection_layers = tuple(int(i) for i in injection_layers)
        self.inject_on_observed = inject_on_observed
        self.use_spatial_prior = use_spatial_prior
        self.spatial_prior_dilation_tokens = int(
            spatial_prior_dilation_tokens)
        if self.spatial_prior_dilation_tokens < 0:
            raise ValueError("spatial_prior_dilation_tokens must be non-negative")
        self.use_velocity = use_velocity
        self.velocity_dim = velocity_dim
        self.validate_inputs = validate_inputs

        self.class_embedding = torch.nn.Embedding(
            num_classes, class_embedding_dim)
        self.box_encoder = torch.nn.Sequential(
            torch.nn.Linear(box_dim, slot_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(slot_dim, slot_dim),
        )
        self.velocity_encoder = (
            torch.nn.Linear(velocity_dim, slot_dim)
            if use_velocity else None
        )
        self.observed_token = torch.nn.Parameter(torch.randn(slot_dim) * 0.02)
        self.missing_token = torch.nn.Parameter(torch.randn(slot_dim) * 0.02)
        self.class_projection = torch.nn.Linear(class_embedding_dim, slot_dim)
        self.scene_projection = torch.nn.Linear(hidden_dim, slot_dim)
        self.input_norm = torch.nn.LayerNorm(slot_dim)
        self.memory_cell = torch.nn.GRUCell(slot_dim, slot_dim)

        # A single object still needs spatially varying influence.  This
        # learned query/key gate avoids the constant attention weight produced
        # by softmax attention over one slot.
        self.query_projection = torch.nn.Linear(hidden_dim, slot_dim)
        self.key_projection = torch.nn.Linear(slot_dim, slot_dim)
        self.value_projection = torch.nn.Linear(slot_dim, slot_dim)
        self.residual_projections = torch.nn.ModuleDict({
            str(i): torch.nn.Linear(slot_dim, hidden_dim)
            for i in self.injection_layers
        })
        for projection in self.residual_projections.values():
            torch.nn.init.zeros_(projection.weight)
            torch.nn.init.zeros_(projection.bias)

        self.last_slot_encoding: Optional[torch.Tensor] = None

    def _validate(
        self,
        availability: torch.Tensor,
        slot_mask: torch.Tensor,
        source_frame_indices: Optional[torch.Tensor],
    ) -> None:
        if not self.validate_inputs or source_frame_indices is None:
            return

        time = availability.shape[1]
        frame_indices = torch.arange(
            time, device=availability.device).view(1, time, 1)
        unavailable = slot_mask & ~availability
        observed = slot_mask & availability
        absent = ~slot_mask
        if torch.any(source_frame_indices[unavailable] >= frame_indices.expand_as(
                source_frame_indices)[unavailable]):
            raise AssertionError(
                "Unavailable slots may only use a strictly earlier observation")
        if torch.any(source_frame_indices[unavailable] < 0):
            raise AssertionError(
                "Unavailable slots require a valid last-observed source frame")
        if torch.any(source_frame_indices[observed] != frame_indices.expand_as(
                source_frame_indices)[observed]):
            raise AssertionError(
                "Observed slots must use their current frame observation")
        if torch.any(source_frame_indices[absent] != -1):
            raise AssertionError("Explicitly absent slots must not carry a box")

    def encode_slots(
        self,
        scene_context: torch.Tensor,
        class_ids: torch.Tensor,
        box_states: torch.Tensor,
        availability: torch.Tensor,
        slot_mask: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        source_frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode slots with a causal recurrent memory.

        Missing frames receive only the dataset-provided last observation,
        previous hidden state, learned missing token, and current scene context.
        Inactive slots are exactly zero and reset the recurrent state.
        """
        availability = availability.bool()
        slot_mask = slot_mask.bool()
        self._validate(availability, slot_mask, source_frame_indices)

        batch, time, slot_count = class_ids.shape
        class_features = self.class_projection(
            self.class_embedding(class_ids.long()))
        box_features = self.box_encoder(box_states)
        scene_features = self.scene_projection(scene_context).unsqueeze(2)
        velocity_features = 0
        if self.use_velocity:
            if velocities is None:
                raise AssertionError(
                    "use_velocity=True requires observed velocity tensors")
            velocity_features = self.velocity_encoder(velocities)

        memory = torch.zeros(
            batch * slot_count, self.slot_dim,
            device=box_states.device, dtype=box_states.dtype)
        outputs = []
        for frame in range(time):
            active = slot_mask[:, frame]
            observed = availability[:, frame] & active
            state_token = torch.where(
                observed.unsqueeze(-1),
                self.observed_token.view(1, 1, -1),
                self.missing_token.view(1, 1, -1),
            )
            update = class_features[:, frame] + box_features[:, frame] + \
                scene_features[:, frame] + state_token
            if self.use_velocity:
                update = update + velocity_features[:, frame]
            update = self.input_norm(update)
            next_memory = self.memory_cell(
                update.reshape(batch * slot_count, self.slot_dim), memory)
            active_flat = active.reshape(batch * slot_count, 1)
            # Explicit absence terminates/removes the slot and its history.
            memory = torch.where(active_flat, next_memory, torch.zeros_like(memory))
            outputs.append(memory.view(batch, slot_count, self.slot_dim))

        encoding = torch.stack(outputs, dim=1)
        self.last_slot_encoding = encoding.detach()
        return encoding

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        view_count: int,
        class_ids: torch.Tensor,
        box_states: torch.Tensor,
        availability: torch.Tensor,
        slot_mask: torch.Tensor,
        layer_index: int,
        velocities: Optional[torch.Tensor] = None,
        source_frame_indices: Optional[torch.Tensor] = None,
        spatial_prior: Optional[torch.Tensor] = None,
        spatial_height: Optional[int] = None,
        spatial_width: Optional[int] = None,
    ) -> torch.Tensor:
        if layer_index not in self.injection_layers:
            return torch.zeros_like(hidden_states)

        spatial_count = hidden_states.shape[1]
        structured = hidden_states.reshape(
            batch_size, sequence_length, view_count,
            spatial_count, self.hidden_dim)
        scene_context = structured.mean(dim=(2, 3))
        slots = self.encode_slots(
            scene_context, class_ids, box_states, availability, slot_mask,
            velocities=velocities,
            source_frame_indices=source_frame_indices)

        queries = self.query_projection(structured)
        keys = self.key_projection(slots).unsqueeze(2).unsqueeze(3)
        values = self.value_projection(slots).unsqueeze(2).unsqueeze(3)
        injection_mask = slot_mask.bool()
        if not self.inject_on_observed:
            # Observed frames update the causal memory but need no corrective
            # residual: the base raster already contains their current box.
            # Restricting injection to unavailable frames makes clean and
            # explicitly absent inputs a hard no-op after training as well as
            # at zero initialization.
            injection_mask = injection_mask & ~availability.bool()
        active = injection_mask.unsqueeze(2).unsqueeze(3).unsqueeze(-1)
        gates = torch.sigmoid(
            (queries.unsqueeze(-2) * keys).sum(-1, keepdim=True) /
            self.slot_dim ** 0.5)
        gated_values = (gates * values * active).sum(dim=-2)
        normalizer = active.sum(dim=-2).clamp_min(1)
        fused = gated_values / normalizer
        residual = self.residual_projections[str(layer_index)](fused)
        # A trained projection can have a non-zero bias.  Without a final
        # presence gate, an explicitly absent (all-slots-inactive) frame would
        # therefore receive a spatially constant residual even though
        # ``fused`` is exactly zero.  Preserve the semantic contract that no
        # active object slot means no adapter intervention.
        frame_has_active_slot = injection_mask.any(dim=2).view(
            batch_size, sequence_length, 1, 1, 1)
        residual = residual * frame_has_active_slot.to(residual.dtype)
        if self.use_spatial_prior:
            if spatial_prior is None or spatial_height is None or \
                    spatial_width is None:
                raise ValueError(
                    "Spatial prior and transformer grid size are required")
            if spatial_height * spatial_width != spatial_count:
                raise ValueError(
                    "Transformer grid does not match spatial token count")
            flat_prior = spatial_prior.reshape(
                -1, 1, *spatial_prior.shape[-2:]).to(residual.dtype)
            if all(i >= j for i, j in zip(
                    flat_prior.shape[-2:], (spatial_height, spatial_width))):
                flat_prior = torch.nn.functional.adaptive_max_pool2d(
                    flat_prior, output_size=(spatial_height, spatial_width))
            else:
                flat_prior = torch.nn.functional.interpolate(
                    flat_prior, size=(spatial_height, spatial_width),
                    mode="nearest")
            if self.spatial_prior_dilation_tokens:
                radius = self.spatial_prior_dilation_tokens
                flat_prior = torch.nn.functional.max_pool2d(
                    flat_prior, kernel_size=2 * radius + 1, stride=1,
                    padding=radius)
            spatial_gate = flat_prior.view(
                batch_size, sequence_length, view_count,
                spatial_count, 1)
            residual = residual * spatial_gate
        return residual.reshape_as(hidden_states)
