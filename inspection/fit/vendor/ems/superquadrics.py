# VENDORED - see ATTRIBUTION.md. Upstream: bmlklwx/EMS-superquadric_fitting
# @5a213d4 (2024-10-03), MIT. Local patches are marked '# PATCH:'.
import numpy as np
from scipy.spatial.transform import Rotation as R
# PATCH: upstream imports EMS.utilities here purely for showSuperquadric(),
# which pulls in mayavi. We do not render from this module.

class superquadric(object):
    # a class object specifies a superquadric primitive with a general pose.
    # attributes:
    # .shape: the shape parameters of a superquadric, an np-arrary of shape 1*2
    #
    #
    #

    def __init__(self, shape_vec, scale_vec, euler_vec, translation):
        self.shape = shape_vec
        self.scale = scale_vec
        self.euler = euler_vec
        self.translation = translation

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        self._shape = np.array(val, dtype=float)

    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, val):
        self._scale = np.array(val, dtype=float)

    @property
    def euler(self):
        return self._r.as_euler('ZYX')

    @euler.setter
    def euler(self, val):
        self._r = R.from_euler('ZYX', val)

    @property
    def translation(self):
        return self._translation
    
    @translation.setter
    def translation(self, val):
        self._translation = np.array(val, dtype=float)

    @property
    def RotM(self):
        return self._r.as_matrix()

    @RotM.setter
    def RotM(self, val):
        self._r = R.from_matrix(val)

    @property
    def quat(self):
        return self._r.as_quat()

    @quat.setter
    def quat(self, val):
        self._r = R.from_quat(val)

    def showSuperquadric(self, threshold = 1e-2, num_limit = 10000, arclength = 0.02):
        # PATCH: mayavi rendering dropped; inspection.fit.render draws these.
        raise NotImplementedError("vendored EMS does not render; see inspection/fit/render.py")


class rotations(object):
    def __init__(self):
        self.euler = [0, 0, 0]

    @property
    def RotM(self):
        return self._r.as_matrix()

    @RotM.setter
    def RotM(self, val):
        self._r = R.from_matrix(val)

    @property
    def quat(self):
        return self._r.as_quat()

    @quat.setter
    def quat(self, val):
        self._r = R.from_quat(val)

    @property
    def euler(self):
        return self._r.as_euler('ZYX')

    @euler.setter
    def euler(self, val):
        self._r = R.from_euler('ZYX', val)