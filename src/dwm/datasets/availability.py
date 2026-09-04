"""Dataset-side construction for object observation availability experiments."""

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageChops, ImageDraw
import torch
import transforms3d

import dwm.datasets.common
from dwm.datasets.nuscenes import MotionDataset


MODE_CLEAN = "clean"
MODE_UNAVAILABLE = "temporary_unavailable"
MODE_ABSENT = "explicit_absent"
MODE_MIXED = "mixed"
MODE_TO_ID = {
    MODE_CLEAN: 0,
    MODE_UNAVAILABLE: 1,
    MODE_ABSENT: 2,
}

DEFAULT_CLASS_PREFIXES = (
    "human.pedestrian",
    "vehicle.bicycle",
    "vehicle.motorcycle",
    "vehicle.bus",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.emergency",
    "vehicle.trailer",
    "vehicle.truck",
    "movable_object.barrier",
    "movable_object.trafficcone",
)


def _stable_int64(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8], "little") & \
        ((1 << 63) - 1)


class ObjectAvailabilityDataset(torch.utils.data.Dataset):
    """Wrap nuScenes ``MotionDataset`` with one persistent selected-object slot.

    The wrapper must sit *inside* ``DatasetAdapter`` so the normal RGB/layout
    transforms still run.  A manifest is recommended for training; without one
    the target and interval are selected deterministically from ``index``.

    The only GT-derived value retained during unavailable frames is
    ``loss_target_roi``.  Its name and separate construction are intentional:
    CTSD may use it to weight the loss, but never passes it to the model.
    """

    def __init__(
        self,
        base_dataset: MotionDataset,
        mode: str = MODE_MIXED,
        mode_weights: Optional[dict] = None,
        missing_durations: tuple = (1, 3, 5, 7),
        min_observed_before: int = 3,
        min_observed_after: int = 2,
        seed: int = 0,
        manifest_path: Optional[str] = None,
        roi_size: tuple = (256, 448),
        strict: bool = True,
        debug_assertions: bool = True,
    ):
        if not isinstance(base_dataset, MotionDataset):
            raise TypeError(
                "ObjectAvailabilityDataset currently requires nuScenes "
                "MotionDataset; PreviewDataset has no self-contained tracks")
        if mode not in (*MODE_TO_ID, MODE_MIXED):
            raise ValueError(f"Unknown object availability mode: {mode}")
        allowed_durations = {1, 3, 5, 7}
        if not set(missing_durations).issubset(allowed_durations):
            raise ValueError(
                f"missing_durations must be a subset of {allowed_durations}")
        if base_dataset._3dbox_image_settings is None:
            raise ValueError("The wrapped dataset must enable 3D-box images")

        self.base_dataset = base_dataset
        self.mode = mode
        self.mode_weights = mode_weights or {
            MODE_CLEAN: 0.5,
            MODE_UNAVAILABLE: 0.4,
            MODE_ABSENT: 0.1,
        }
        if set(self.mode_weights) != set(MODE_TO_ID):
            raise ValueError("mode_weights must define clean/unavailable/absent")
        self.missing_durations = tuple(int(i) for i in missing_durations)
        self.min_observed_before = int(min_observed_before)
        self.min_observed_after = int(min_observed_after)
        self.seed = int(seed)
        self.roi_size = tuple(int(i) for i in roi_size)
        self.strict = strict
        self.debug_assertions = debug_assertions

        self.manifest_records = None
        if manifest_path is not None:
            payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            self.manifest_records = payload.get("records", payload)
            if not isinstance(self.manifest_records, list):
                raise ValueError("availability manifest records must be a list")

    def __len__(self):
        return len(self.manifest_records) if self.manifest_records is not None \
            else len(self.base_dataset)

    def _base_index_and_record(self, index: int):
        if self.manifest_records is not None:
            record = self.manifest_records[index]
            return int(record["base_index"]), record
        return index, self.find_record(index)

    def _segment(self, base_index: int):
        item = self.base_dataset.items[base_index]
        return [
            [
                MotionDataset.query(
                    self.base_dataset.tables, self.base_dataset.indices,
                    "sample_data", token)
                for token in frame
            ]
            for frame in item["segment"]
        ]

    def _frame_annotations(self, segment: list):
        output = []
        for frame in segment:
            camera = next(
                sample_data for sample_data in frame
                if MotionDataset.check_sensor(
                    self.base_dataset.tables, self.base_dataset.indices,
                    sample_data, modality="camera"))
            annotations = MotionDataset.query_range(
                self.base_dataset.tables, self.base_dataset.indices,
                "sample_annotation", camera["sample_token"],
                column_name="sample_token")
            output.append({i["instance_token"]: i for i in annotations})
        return output

    def _category_name(self, instance_token: str) -> str:
        instance = MotionDataset.query(
            self.base_dataset.tables, self.base_dataset.indices,
            "instance", instance_token)
        category = MotionDataset.query(
            self.base_dataset.tables, self.base_dataset.indices,
            "category", instance["category_token"])
        return category["name"]

    def _class_id(self, category_name: str) -> int:
        for index, prefix in enumerate(DEFAULT_CLASS_PREFIXES, start=1):
            if category_name.startswith(prefix):
                return index
        return 0

    def _changes_raster(
        self, segment_frame: list, target_instance_token: str
    ) -> bool:
        """Return true only when deleting the target changes a camera raster."""
        for sample_data in segment_frame:
            if not MotionDataset.check_sensor(
                    self.base_dataset.tables, self.base_dataset.indices,
                    sample_data, modality="camera"):
                continue
            original = MotionDataset.get_3dbox_image(
                self.base_dataset.tables, self.base_dataset.indices,
                sample_data, self.base_dataset._3dbox_image_settings)
            removed = MotionDataset.get_3dbox_image(
                self.base_dataset.tables, self.base_dataset.indices,
                sample_data, self.base_dataset._3dbox_image_settings,
                excluded_instance_tokens={target_instance_token})
            if ImageChops.difference(original, removed).getbbox() is not None:
                return True
        return False

    def find_record(
        self, base_index: int, duration: Optional[int] = None,
        missing_start: Optional[int] = None,
        target_instance_token: Optional[str] = None,
    ) -> Optional[dict]:
        """Find a deterministic real track satisfying 3-before/2-after."""
        segment = self._segment(base_index)
        annotations = self._frame_annotations(segment)
        time = len(annotations)
        rng = random.Random(self.seed + base_index * 1000003)
        durations = [duration] if duration is not None \
            else list(self.missing_durations)
        rng.shuffle(durations)

        for selected_duration in durations:
            last_start = time - selected_duration - self.min_observed_after
            starts = [missing_start] if missing_start is not None else list(
                range(self.min_observed_before, last_start + 1))
            rng.shuffle(starts)
            for start in starts:
                if start is None or start < self.min_observed_before or \
                        start + selected_duration + self.min_observed_after > time:
                    continue
                required = range(
                    start - self.min_observed_before,
                    start + selected_duration + self.min_observed_after)
                candidates = set(annotations[next(iter(required))])
                for frame in required:
                    candidates.intersection_update(annotations[frame])
                candidates = sorted(
                    token for token in candidates
                    if self._class_id(self._category_name(token)) > 0)
                if target_instance_token is not None:
                    candidates = [
                        token for token in candidates
                        if token == target_instance_token]
                # A selected target must affect at least one configured raster
                # in every missing frame; otherwise deletion is a vacuous no-op.
                candidates = [
                    token for token in candidates
                    if all(self._changes_raster(segment[frame], token)
                           for frame in range(start, start + selected_duration))
                ]
                if not candidates:
                    continue
                target = candidates[rng.randrange(len(candidates))]
                category = self._category_name(target)
                return {
                    "base_index": int(base_index),
                    "target_instance_token": target,
                    "target_instance_hash": _stable_int64(target),
                    "target_category": category,
                    "target_class_id": self._class_id(category),
                    "missing_start": int(start),
                    "missing_duration": int(selected_duration),
                    "missing_end_exclusive": int(start + selected_duration),
                    "min_observed_before": self.min_observed_before,
                    "min_observed_after": self.min_observed_after,
                    "pair_seed": self.seed + base_index * 1000003,
                }
        return None

    def _resolve_mode(self, base_index: int) -> str:
        if self.mode != MODE_MIXED:
            return self.mode
        rng = random.Random(self.seed + base_index * 9176 + 41)
        modes = list(MODE_TO_ID)
        weights = [self.mode_weights[i] for i in modes]
        return rng.choices(modes, weights=weights, k=1)[0]

    @staticmethod
    def _normalized_box_state(annotation: dict, reference_from_world) -> torch.Tensor:
        world_from_annotation = dwm.datasets.common.get_transform(
            annotation["rotation"], annotation["translation"])
        reference_from_annotation = reference_from_world @ world_from_annotation
        center = reference_from_annotation[:3, 3] / np.array([50.0, 50.0, 10.0])
        size = np.asarray(annotation["size"], dtype=np.float64) / \
            np.array([10.0, 10.0, 5.0])
        quaternion = transforms3d.quaternions.mat2quat(
            reference_from_annotation[:3, :3])
        return torch.tensor(
            np.concatenate([center, size, quaternion]), dtype=torch.float32)

    def _make_loss_roi(
        self, sample_data: dict, annotation: Optional[dict]
    ) -> torch.Tensor:
        height, width = self.roi_size
        image = Image.new("L", (width, height), 0)
        if annotation is None:
            return torch.zeros((height, width), dtype=torch.uint8)

        calibrated_sensor = MotionDataset.query(
            self.base_dataset.tables, self.base_dataset.indices,
            "calibrated_sensor", sample_data["calibrated_sensor_token"])
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = np.asarray(calibrated_sensor["camera_intrinsic"])
        ego_from_camera = dwm.datasets.common.get_transform(
            calibrated_sensor["rotation"], calibrated_sensor["translation"])
        world_from_ego = dwm.datasets.common.get_transform(
            sample_data["rotation"], sample_data["translation"])
        image_from_world = intrinsic @ np.linalg.inv(
            world_from_ego @ ego_from_camera)
        scale = np.diag([
            annotation["size"][1], annotation["size"][0],
            annotation["size"][2], 1])
        world_from_annotation = dwm.datasets.common.get_transform(
            annotation["rotation"], annotation["translation"])
        corners = image_from_world @ world_from_annotation @ scale @ \
            np.asarray(MotionDataset.default_3dbox_corner_template).T

        projected = []
        for a, b in MotionDataset.default_3dbox_edge_indices:
            line = dwm.datasets.common.project_line(corners[:, a], corners[:, b])
            if line is not None:
                projected.extend([(line[0], line[1]), (line[2], line[3])])
        if projected:
            xs, ys = zip(*projected)
            x_scale = width / sample_data["width"]
            y_scale = height / sample_data["height"]
            box = (
                max(0, min(width, min(xs) * x_scale)),
                max(0, min(height, min(ys) * y_scale)),
                max(0, min(width, max(xs) * x_scale)),
                max(0, min(height, max(ys) * y_scale)),
            )
            if box[2] > box[0] and box[3] > box[1]:
                ImageDraw.Draw(image).rectangle(box, fill=1)
        return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())

    def _build_model_inputs(
        self, segment: list, annotations: list, record: dict, mode: str
    ) -> dict:
        time = len(segment)
        target = record["target_instance_token"]
        start = record["missing_start"]
        stop = record["missing_end_exclusive"]
        reference_world_from_ego = dwm.datasets.common.get_transform(
            segment[0][0]["rotation"], segment[0][0]["translation"])
        reference_from_world = np.linalg.inv(reference_world_from_ego)

        class_ids = torch.zeros((time, 1), dtype=torch.long)
        box_states = torch.zeros((time, 1, 10), dtype=torch.float32)
        availability = torch.zeros((time, 1), dtype=torch.bool)
        slot_mask = torch.zeros((time, 1), dtype=torch.bool)
        source_frames = torch.full((time, 1), -1, dtype=torch.long)
        state_codes = torch.zeros((time, 1), dtype=torch.long)
        last_observed_frame = None

        for frame in range(time):
            annotation = annotations[frame].get(target)
            in_interval = start <= frame < stop
            if mode == MODE_ABSENT and in_interval:
                # Explicit absence removes and terminates the slot.
                last_observed_frame = None
                continue

            if mode == MODE_UNAVAILABLE and in_interval:
                if last_observed_frame is None:
                    raise AssertionError(
                        "Unavailable interval has no prior observed state")
                source_annotation = annotations[last_observed_frame][target]
                class_ids[frame, 0] = record["target_class_id"]
                box_states[frame, 0] = self._normalized_box_state(
                    source_annotation, reference_from_world)
                availability[frame, 0] = False
                slot_mask[frame, 0] = True
                source_frames[frame, 0] = last_observed_frame
                state_codes[frame, 0] = 1
                continue

            if annotation is not None:
                class_ids[frame, 0] = record["target_class_id"]
                box_states[frame, 0] = self._normalized_box_state(
                    annotation, reference_from_world)
                availability[frame, 0] = True
                slot_mask[frame, 0] = True
                source_frames[frame, 0] = frame
                state_codes[frame, 0] = 2
                last_observed_frame = frame
            else:
                last_observed_frame = None

        # Leakage assertions concern model inputs.  The GT ROI is constructed
        # separately and is never returned by CTSD.get_conditions().
        if self.debug_assertions:
            interval = slice(start, stop)
            if mode == MODE_UNAVAILABLE:
                expected_source = start - 1
                assert torch.all(~availability[interval])
                assert torch.all(slot_mask[interval])
                assert torch.all(source_frames[interval] == expected_source)
                expected_box = box_states[expected_source:expected_source + 1]
                assert torch.equal(
                    box_states[interval], expected_box.expand_as(box_states[interval]))
            elif mode == MODE_ABSENT:
                assert torch.all(~slot_mask[interval])
                assert torch.all(source_frames[interval] == -1)
                assert torch.count_nonzero(box_states[interval]) == 0

        return {
            "object_class_ids": class_ids,
            "object_box_states": box_states,
            "object_availability": availability,
            "object_slot_mask": slot_mask,
            "object_source_frame_indices": source_frames,
            "object_state_codes": state_codes,
            "object_track_hash": torch.tensor(
                [record["target_instance_hash"]], dtype=torch.long),
            "object_availability_mode_id": torch.tensor(
                MODE_TO_ID[mode], dtype=torch.long),
            "object_missing_interval": torch.tensor([start, stop]),
            "object_pair_seed": torch.tensor(record["pair_seed"], dtype=torch.long),
        }

    def _apply_mode(
        self, base_item: dict, base_index: int, record: dict, mode: str
    ) -> dict:
        segment = self._segment(base_index)
        annotations = self._frame_annotations(segment)
        target = record["target_instance_token"]
        start = record["missing_start"]
        stop = record["missing_end_exclusive"]

        if self.debug_assertions:
            required = range(
                start - self.min_observed_before,
                stop + self.min_observed_after)
            assert all(target in annotations[frame] for frame in required)

        result = copy.deepcopy(base_item)
        if mode in (MODE_UNAVAILABLE, MODE_ABSENT):
            for frame in range(start, stop):
                result["3dbox_images"][frame] = [
                    MotionDataset.get_3dbox_image(
                        self.base_dataset.tables, self.base_dataset.indices,
                        sample_data, self.base_dataset._3dbox_image_settings,
                        excluded_instance_tokens={target})
                    for sample_data in segment[frame]
                    if MotionDataset.check_sensor(
                        self.base_dataset.tables, self.base_dataset.indices,
                        sample_data, modality="camera")
                ]

        result.update(self._build_model_inputs(
            segment, annotations, record, mode))
        result["loss_target_roi"] = torch.stack([
            torch.stack([
                self._make_loss_roi(sample_data, annotations[frame].get(target))
                for sample_data in segment[frame]
                if MotionDataset.check_sensor(
                    self.base_dataset.tables, self.base_dataset.indices,
                    sample_data, modality="camera")
            ])
            for frame in range(len(segment))
        ])
        return result

    def make_paired_items(
        self, index: int, base_item: Optional[dict] = None
    ) -> tuple[dict, dict, dict, dict]:
        """Build A/B/C from one base item for invariance auditing."""
        base_index, record = self._base_index_and_record(index)
        if record is None:
            raise RuntimeError(f"No eligible persistent object at index {base_index}")
        if base_item is None:
            base_item = self.base_dataset[base_index]
        return (
            self._apply_mode(base_item, base_index, record, MODE_CLEAN),
            self._apply_mode(base_item, base_index, record, MODE_UNAVAILABLE),
            self._apply_mode(base_item, base_index, record, MODE_ABSENT),
            record,
        )

    def __getitem__(self, index: int):
        base_index, record = self._base_index_and_record(index)
        if record is None:
            if self.strict:
                raise RuntimeError(
                    f"No eligible persistent object at index {base_index}; "
                    "build and use an eligibility manifest")
            return self.base_dataset[base_index]
        mode = self._resolve_mode(base_index)
        return self._apply_mode(
            self.base_dataset[base_index], base_index, record, mode)
