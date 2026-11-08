import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from delta_robot.pick.scene import augment_scene


class SceneAugmentationTest(unittest.TestCase):
    def test_adds_pool_bin_and_welds_without_removing_robot(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "scene.xml"
            source.write_text('''<mujoco><worldbody><body name="base"/><body name="old_part"/></worldbody><equality/><contact><exclude body1="end_effector" body2="old_part"/></contact></mujoco>''')
            cad = tmp / "cad.json"
            cad.write_text(json.dumps({
                "conveyor": {"world_position_mm": [0, 0, 0], "length": 800, "belt_width": 200, "belt_thickness": 10},
                "scene": {"floor_z_m": -0.5, "object_gap_m": 0.002},
                "objects": [{"name": "old_part"}]
            }))
            runtime = {
                "pool_size": 3,
                "object": {"radius_m": 0.02, "height_m": 0.04, "mass_kg": 0.05, "friction": [0.8, 0.01, 0.001]},
                "bin": {"position_xy_m": [0, 0.25], "inner_half_size_m": [0.1, 0.08], "wall_height_m": 0.1, "wall_thickness_m": 0.01},
                "controller": {"gripper_body_name": "gripper"}
            }
            output = tmp / "scene_multi.xml"
            augment_scene(source, output, cad, runtime)
            xml = output.read_text()
            self.assertIn('name="base"', xml)
            self.assertNotIn('name="old_part"', xml)
            self.assertIn('name="pick_object_000"', xml)
            self.assertIn('name="grasp_pick_object_002"', xml)
            self.assertIn('name="collection_bin"', xml)
            self.assertNotIn('body2="old_part"', xml)


if __name__ == "__main__":
    unittest.main()
