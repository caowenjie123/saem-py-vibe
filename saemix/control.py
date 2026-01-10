from typing import Any, Dict, Optional, Tuple

import numpy as np


def saemix_control(
    map: bool = True,
    fim: bool = True,
    ll_is: bool = False,
    ll_gq: bool = False,
    nbiter_saemix: Tuple[int, int] = (300, 100),
    nbiter_sa: Optional[int] = None,
    nb_chains: int = 1,
    fix_seed: bool = True,
    seed: Optional[int] = 23456,
    rng: Optional[np.random.Generator] = None,
    nmc_is: int = 5000,
    nu_is: int = 4,
    print_is: bool = False,
    nbdisplay: int = 100,
    display_progress: bool = False,
    nbiter_burn: int = 5,
    nbiter_map: int = 5,
    nbiter_mcmc: Tuple[int, int, int, int] = (2, 2, 2, 0),
    proba_mcmc: float = 0.4,
    stepsize_rw: float = 0.4,
    rw_init: float = 0.5,
    alpha_sa: float = 0.97,
    nnodes_gq: int = 12,
    nsd_gq: int = 4,
    maxim_maxiter: int = 100,
    nb_sim: int = 1000,
    nb_simpred: int = 100,
    ipar_lmcmc: int = 50,
    ipar_rmcmc: float = 0.05,
    print_results: bool = True,
    save: bool = True,
    save_graphs: bool = True,
    directory: str = "newdir",
    warnings: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create SAEM control parameters.

    Parameters
    ----------
    seed : int, optional
        Random seed for creating RNG (default: 23456)
    rng : numpy.random.Generator, optional
        User-provided RNG instance. If provided, seed is ignored.
    fix_seed : bool
        If False and rng is None, a random seed will be generated.

    Returns
    -------
    Dict[str, Any]
        Control parameters dictionary including 'rng' key with Generator instance.
    """
    if ipar_lmcmc < 2:
        ipar_lmcmc = 2
        if warnings:
            print("Value of L_MCMC too small, setting it to 2")

    if nbiter_sa is None:
        nbiter_sa = nbiter_saemix[0] // 2
    elif nbiter_sa > nbiter_saemix[0]:
        if warnings:
            print(
                "The number of iterations for simulated annealing should be <= K1, setting it to nbiter_saemix[0]"
            )
        nbiter_sa = nbiter_saemix[0]

    # Create RNG instance - unified random number management
    # Priority: user-provided rng > seed > random seed (if fix_seed=False)
    if rng is not None:
        _rng = rng
    elif seed is not None and fix_seed:
        _rng = np.random.default_rng(seed)
    elif not fix_seed:
        # Generate a random seed without using global state
        _rng = np.random.default_rng()
        seed = int(_rng.integers(1, 2**31 - 1))
        _rng = np.random.default_rng(seed)
    else:
        _rng = np.random.default_rng(seed)

    control = {
        "map": map,
        "fim": fim,
        "ll_is": ll_is,
        "ll_gq": ll_gq,
        "nbiter_saemix": nbiter_saemix,
        "nbiter_sa": nbiter_sa,
        "nbiter_burn": nbiter_burn,
        "nbiter_map": nbiter_map,
        "nb_chains": nb_chains,
        "fix_seed": fix_seed,
        "seed": seed,
        "rng": _rng,
        "nmc_is": nmc_is,
        "nu_is": nu_is,
        "print_is": print_is,
        "nbdisplay": nbdisplay,
        "display_progress": display_progress,
        "print": print_results,
        "save": save,
        "save_graphs": save_graphs,
        "directory": directory,
        "warnings": warnings,
        "nbiter_mcmc": nbiter_mcmc,
        "proba_mcmc": proba_mcmc,
        "stepsize_rw": stepsize_rw,
        "rw_init": rw_init,
        "alpha_sa": alpha_sa,
        "nnodes_gq": nnodes_gq,
        "nsd_gq": nsd_gq,
        "maxim_maxiter": maxim_maxiter,
        "nb_sim": nb_sim,
        "nb_simpred": nb_simpred,
        "ipar_lmcmc": ipar_lmcmc,
        "ipar_rmcmc": ipar_rmcmc,
    }

    control.update(kwargs)
    control["nbiter_tot"] = sum(nbiter_saemix)

    return control
