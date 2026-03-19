"""
Large Magellanic Cloud as a time-dependent gravitational perturbation.

The LMC is modeled as a Hernquist sphere on a backward-integrated orbit
with dynamical friction. At each timestep, it contributes a gravitational
potential centered on its orbital position.

Parameters from Erkal+2019 (Orphan stream fit):
    M_LMC = 1.38 x 10^11 Msun
    a_LMC = 14.9 kpc (Hernquist scale radius)

Present-day position/velocity from galpy's Orbit.from_name('LMC'),
which draws on Kallivayalil+2013 proper motions and Pietrzynski+2013 distance.

The LMC orbit is integrated backward in the MW potential WITH dynamical
friction (ChandrasekharDynamicalFrictionForce), then wrapped in
MovingObjectPotential for use in stream orbit integration.

Reference:
    Erkal et al. 2019, MNRAS, 487, 2685
    Vasiliev, Belokurov & Erkal 2021, MNRAS, 501, 2279
"""

import numpy as np
from galpy.orbit import Orbit
from galpy.potential import (
    HernquistPotential,
    MovingObjectPotential,
    ChandrasekharDynamicalFrictionForce,
)
from galpy.util.conversion import time_in_Gyr

# Solar parameters (must match everywhere)
RO = 8.122   # kpc
VO = 229.0   # km/s

# LMC parameters (Erkal+2019)
M_LMC_MSUN = 1.38e11              # total mass in solar masses
A_LMC_KPC = 14.9                  # Hernquist scale radius in kpc

# Convert to galpy natural units
# Natural mass unit = vo^2 * ro / G
# For Hernquist: amp = 2 * M_total (galpy convention)
from galpy.util.conversion import mass_in_msol
_MASS_UNIT = mass_in_msol(VO, RO)  # 1 natural mass unit in Msun
A_LMC_NAT = A_LMC_KPC / RO
AMP_LMC_NAT = 2.0 * M_LMC_MSUN / _MASS_UNIT  # factor of 2 for Hernquist convention


def build_lmc_potential(mw_pot, t_back_gyr=3.0, n_steps=3000):
    """
    Build the LMC as a MovingObjectPotential.

    Steps:
    1. Initialize LMC at present-day position from Gaia/literature
    2. Set up dynamical friction from the MW halo
    3. Integrate LMC orbit backward t_back_gyr
    4. Wrap in MovingObjectPotential with Hernquist profile

    Parameters
    ----------
    mw_pot : list of galpy Potential objects
        The MW potential (without LMC) — needed for dynamical friction
        density and for integrating the LMC orbit.
    t_back_gyr : float
        How far back to integrate the LMC orbit (Gyr).
    n_steps : int
        Number of integration steps.

    Returns
    -------
    lmc_moving : MovingObjectPotential
        Time-dependent LMC potential to add to the MW potential list.
    lmc_orbit : Orbit
        The integrated LMC orbit (for diagnostics).
    """
    # LMC present-day phase space in galpy natural units.
    # Hardcoded from Orbit.from_name('LMC') with ro=8.122, vo=229,
    # zo=0.0208, solarmotion='schoenrich'. Avoids external lookup
    # for reproducibility on offline/cluster nodes.
    # Source: Gaia DR3 + Kallivayalil+2013 + Pietrzynski+2013
    lmc = Orbit([5.0553000404,    # R (natural)
                 0.9992417185,    # vR (natural)
                 0.2221253254,    # vT (natural)
                 -3.4283697540,   # z (natural)
                 0.9121802587,    # vz (natural)
                 -1.5433169752])  # phi (radians)

    # Hernquist profile for the LMC mass distribution
    lmc_hernquist = HernquistPotential(amp=AMP_LMC_NAT, a=A_LMC_NAT)

    # Dynamical friction: the MW halo decelerates the LMC
    # GMs = G * M_LMC in natural units = AMP_LMC_NAT / 2 (undo the factor of 2)
    # rhm = half-mass radius of Hernquist = (1 + sqrt(2)) * a
    rhm_nat = (1.0 + np.sqrt(2.0)) * A_LMC_NAT

    cdf = ChandrasekharDynamicalFrictionForce(
        GMs=AMP_LMC_NAT / 2.0,
        rhm=rhm_nat,
        dens=mw_pot,
    )

    # Combine MW potential + dynamical friction for LMC orbit integration
    pot_for_lmc = mw_pot + [cdf]

    # Time array: 0 to -t_back_gyr in natural units
    t_nat_max = t_back_gyr / time_in_Gyr(VO, RO)
    ts = np.linspace(0, -t_nat_max, n_steps)

    # Integrate LMC orbit backward
    lmc.integrate(ts, pot_for_lmc)

    # Wrap in MovingObjectPotential
    lmc_moving = MovingObjectPotential(
        lmc, pot=lmc_hernquist,
    )

    return lmc_moving, lmc
