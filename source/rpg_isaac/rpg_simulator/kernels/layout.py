"""Fixed layout of the 33-dim quad state vector and the 6-dim body wrench.

Kernels bake these as compile-time constants; ``QuadState`` asserts it uses
the same layout. Everything that varies per instance (airframe, controller
gains, timing) is passed to the kernels at runtime instead.
"""

POS = 0  # 3
ATT = 3  # 4, wxyz
VEL = 7  # 3, world frame
OME = 10  # 3, body frame
ACC = 13  # 3
TAU = 16  # 3
MOT = 19  # 4, motor speeds
MOTDES = 23  # 4, desired motor speeds
E = 27  # energy
UCAP = 28  # capacitor voltage
UBAT = 29  # battery voltage
BFLI = 30  # 3, betaflight integrator state
STATE_DIM = 33

# force_torque tensor [fx, fy, fz, tx, ty, tz]
FX, FY, FZ, TX, TY, TZ = range(6)
WRENCH_DIM = 6
