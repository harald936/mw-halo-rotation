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

# LMC present-day phase space in galpy natural units.
# Hardcoded from Orbit.from_name('LMC') with ro=8.122, vo=229,
# zo=0.0208, solarmotion='schoenrich'. Avoids external lookup
# for reproducibility on offline/cluster nodes.
# Source: Gaia DR3 + Kallivayalil+2013 + Pietrzynski+2013
LMC_VXVV = [
    5.0553000404,    # R (natural)
    0.9992417185,    # vR (natural)
    0.2221253254,    # vT (natural)
    -3.4283697540,   # z (natural)
    0.9121802587,    # vz (natural)
    -1.5433169752,   # phi (radians)
]


def _density_proxy_potential(mw_pot):
    """Return a density proxy list compatible with Chandrasekhar friction.

    SolidBodyRotationWrapperPotential has force support in galpy's C layer, but
    it does not expose density support there. For the friction term only, we can
    therefore optionally use the underlying static triaxial halo density while
    still integrating the LMC in the full rotating halo.
    """
    proxy = list(mw_pot)
    if proxy and hasattr(proxy[-1], "_pot") and not getattr(proxy[-1], "hasC_dens", True):
        proxy[-1] = proxy[-1]._pot
    return proxy


def _build_lmc_orbit(
    mw_pot,
    *,
    t_back_gyr,
    n_steps,
    df_density_pot=None,
    cdf_nr=501,
    integrate_method=None,
):
    """Integrate the backward LMC orbit and return the Orbit instance."""
    lmc = Orbit(LMC_VXVV)

    rhm_nat = (1.0 + np.sqrt(2.0)) * A_LMC_NAT
    dens_pot = mw_pot if df_density_pot is None else df_density_pot

    cdf = ChandrasekharDynamicalFrictionForce(
        GMs=AMP_LMC_NAT / 2.0,
        rhm=rhm_nat,
        dens=dens_pot,
        nr=cdf_nr,
    )

    pot_for_lmc = list(mw_pot) + [cdf]
    t_nat_max = t_back_gyr / time_in_Gyr(VO, RO)
    ts = np.linspace(0, -t_nat_max, n_steps)

    if integrate_method is None:
        lmc.integrate(ts, pot_for_lmc)
    else:
        lmc.integrate(ts, pot_for_lmc, method=integrate_method)

    return lmc


def build_lmc_potential(
    mw_pot,
    t_back_gyr=3.0,
    n_steps=3000,
    *,
    df_density_mode="exact",
    cdf_nr=501,
    integrate_method=None,
):
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
    df_density_mode : {'exact', 'static_proxy'}
        How to evaluate the host density inside Chandrasekhar dynamical
        friction. 'exact' uses the supplied MW potential directly.
        'static_proxy' unwraps the rotating halo for the density lookup only,
        which restores galpy's C integrator for this force combination while
        leaving the actual gravitational forces unchanged.
    cdf_nr : int
        Number of radii used to tabulate sigma_r in the Chandrasekhar force.
    integrate_method : str or None
        Optional galpy Orbit.integrate method override.

    Returns
    -------
    lmc_moving : MovingObjectPotential
        Time-dependent LMC potential to add to the MW potential list.
    lmc_orbit : Orbit
        The integrated LMC orbit (for diagnostics).
    """
    # Hernquist profile for the LMC mass distribution
    lmc_hernquist = HernquistPotential(amp=AMP_LMC_NAT, a=A_LMC_NAT)

    if df_density_mode == "exact":
        df_density_pot = None
    elif df_density_mode == "static_proxy":
        df_density_pot = _density_proxy_potential(mw_pot)
    else:
        raise ValueError(f"Unknown df_density_mode={df_density_mode!r}")

    lmc = _build_lmc_orbit(
        mw_pot,
        t_back_gyr=t_back_gyr,
        n_steps=n_steps,
        df_density_pot=df_density_pot,
        cdf_nr=cdf_nr,
        integrate_method=integrate_method,
    )

    # Wrap in MovingObjectPotential
    lmc_moving = MovingObjectPotential(
        lmc, pot=lmc_hernquist,
    )

    return lmc_moving, lmc


def build_lmc_potential_fast(mw_pot, t_back_gyr=3.0, n_steps=3000, *, cdf_nr=101):
    """Fast next-run LMC builder for sampling.

    This keeps the full rotating MW force for the LMC orbit, but evaluates the
    Chandrasekhar friction density using the underlying static triaxial halo.
    In the current galpy setup this restores the C Dormand-Prince integrator
    and, with a smaller sigma_r interpolation grid, reduces the LMC build cost
    substantially while leaving the integrated orbit very close to the exact
    path.
    """
    return build_lmc_potential(
        mw_pot,
        t_back_gyr=t_back_gyr,
        n_steps=n_steps,
        df_density_mode="static_proxy",
        cdf_nr=cdf_nr,
        integrate_method="dopr54_c",
    )
