"""Saved camera roles: re-matching by stable device ID. Run: python gazecore/tests/test_camera_setup.py"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gazecore.camera import camera_setup as cs  # noqa: E402

EYE_A = r"USB\VID_0C45&PID_6366&MI_00\8&6F2A11A&0&0000"
EYE_B = r"USB\VID_0C45&PID_6366&MI_00\8&183AF011&0&0000"
FRONT = r"USB\VID_0BDA&PID_5697&MI_00\7&1EA1D4FB&0&0000"
DEVICES = [{"index": 0, "device_id": EYE_A, "name": "USB Camera"},
           {"index": 1, "device_id": EYE_B, "name": "USB Camera"},
           {"index": 2, "device_id": FRONT, "name": "USB 2.0 Camera"}]


def test_resolve_by_device_id_not_index():
    binding = cs.RoleBinding(EYE_B, index=1)
    assert cs.resolve(binding, DEVICES) == 1
    assert cs.resolve(binding, [dict(DEVICES[1], index=0)]) == 0  # Renumbered
    assert cs.resolve(binding, [DEVICES[0], DEVICES[2]]) is None  # Unplugged: never index 1's new owner
    assert cs.resolve(binding, None) == 1  # Cannot enumerate: trust the index
    assert cs.resolve(cs.RoleBinding("index:4", index=4), DEVICES) == 4


def test_moved_port_candidates_are_same_model_and_unclaimed():
    gone = cs.RoleBinding(r"USB\VID_0C45&PID_6366&MI_00\9&AAAA&0&0000")
    found = cs.moved_port_candidates(gone, DEVICES, claimed_ids=[EYE_A, FRONT])
    assert [d["device_id"] for d in found] == [EYE_B]


def test_round_trip_and_defaults():
    setup = cs.default_setup(DEVICES)
    assert setup.roles["left"].device_id == EYE_A and setup.roles["left"].flip and not setup.roles["front"].flip
    setup.roles["right"].native_size = [640, 480]
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "settings", "camera_setup.json")
        assert cs.load_setup(path).bound() == {}
        cs.save_setup(path, setup)
        loaded = cs.load_setup(path)
    assert loaded.roles == setup.roles and loaded.saved_at
    assert loaded.roles["right"].geometry == {"device_id": EYE_B, "native_size": [640, 480], "flip": True, "mirror": False}


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("All camera setup tests passed.")
