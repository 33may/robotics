from scipy.spatial.transforms import Rotation

def xyz_to_quat_isaac(x, y, z):
    r = Rotation.from_euler('xyz', [x, y, z], degree=True)

    return (r[3], r[0], r[1], r[2])