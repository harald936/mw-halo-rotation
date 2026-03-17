"""
Fixed baryonic potential components (disk + bulge).

Uses MWPotential2014-style parameterization (Bovy 2015) with
Miyamoto-Nagai disk and power-law bulge with cutoff.

These are held fixed during MCMC — only the halo parameters vary.

The normalize values set each component's fractional contribution
to v_circ^2 at R=R0: disk=60%, bulge=5%, leaving 35% for the halo.

IMPORTANT: Constructed WITHOUT ro/vo kwargs — pure natural units.
Convert inputs/outputs using RO and VO constants.
"""

from galpy.potential import (
    MiyamotoNagaiPotential,
    PowerSphericalPotentialwCutoff,
)

# Solar parameters (must match everywhere in the pipeline)
RO = 8.122   # kpc, Gravity Collaboration 2018 (= Eilers+2019)
VO = 229.0   # km/s, Eilers+2019

# -----------------------------------------------------------------------
# Disk: Miyamoto-Nagai
# -----------------------------------------------------------------------
# Physical: a_d = 3.0 kpc, b_d = 0.28 kpc
# Contributes ~60% of v_circ^2 at R0
DISK_A = 3.0 / RO       # scale length in natural units
DISK_B = 0.28 / RO      # scale height in natural units
DISK_NORMALIZE = 0.6     # fraction of v_circ^2(R0) from disk

# -----------------------------------------------------------------------
# Bulge: Power-law spherical with exponential cutoff
# -----------------------------------------------------------------------
# Physical: alpha=1.8, r_cut = 1.9 kpc
# Contributes ~5% of v_circ^2 at R0
BULGE_ALPHA = 1.8
BULGE_RC = 1.9 / RO     # cutoff radius in natural units
BULGE_NORMALIZE = 0.05   # fraction of v_circ^2(R0) from bulge


def build_baryonic_potential():
    """
    Return list of fixed baryonic potential components [disk, bulge].

    These use galpy natural units (no ro/vo attached).
    The normalize values ensure that disk+bulge contribute 65% of
    v_circ^2 at R=R0, leaving 35% for the dark matter halo.
    """
    disk = MiyamotoNagaiPotential(
        a=DISK_A,
        b=DISK_B,
        normalize=DISK_NORMALIZE,
    )

    bulge = PowerSphericalPotentialwCutoff(
        alpha=BULGE_ALPHA,
        rc=BULGE_RC,
        normalize=BULGE_NORMALIZE,
    )

    return [disk, bulge]
