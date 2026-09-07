import unittest
from collections import defaultdict

from dwm.tools.build_object_availability_balanced_manifest import \
    choose_balanced


class ObjectAvailabilityManifestTest(unittest.TestCase):
    def test_selection_falls_back_from_vacuous_candidate(self):
        scene = "scene"
        duration = 1
        candidates = defaultdict(list)
        candidates[(scene, duration)] = [
            {"base_index": 2, "target_instance_token": "vacuous"},
            {"base_index": 3, "target_instance_token": "valid"},
        ]

        selected = choose_balanced(
            candidates, [scene], [duration], 1,
            record_validator=lambda record: (
                record["target_instance_token"] != "vacuous"))

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["target_instance_token"], "valid")


if __name__ == "__main__":
    unittest.main()
