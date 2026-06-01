import unittest

from view_cameras import parse_args
from cameras import DEFAULT_PRESET


class ViewCamerasTest(unittest.TestCase):
    def test_uses_default_camera_preset(self):
        self.assertEqual(parse_args([]).preset, DEFAULT_PRESET)

    def test_accepts_shared_camera_preset(self):
        self.assertEqual(parse_args(["--preset", "realsense_depth"]).preset, "realsense_depth")


if __name__ == "__main__":
    unittest.main()
