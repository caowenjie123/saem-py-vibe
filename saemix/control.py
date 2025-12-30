from typing import Tuple, Optional, Dict, Any
import random


def saemix_control(
    map: bool = True,
    fim: bool = True,
    ll_is: bool = False,
    ll_gq: bool = False,
    nbiter_saemix: Tuple[int, int] = (300, 100),
    nbiter_sa: Optional[int] = None,
    nb_chains: int = 1,
    fix_seed: bool = True,
    seed: int = 23456,
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
    **kwargs
) -> Dict[str, Any]:
    if ipar_lmcmc < 2:
        ipar_lmcmc = 2
        if warnings:
            print("Value of L_MCMC too small, setting it to 2")
    
    if nbiter_sa is None:
        nbiter_sa = nbiter_saemix[0] // 2
    elif nbiter_sa > nbiter_saemix[0]:
        if warnings:
            print("The number of iterations for simulated annealing should be <= K1, setting it to nbiter_saemix[0]")
        nbiter_sa = nbiter_saemix[0]
    
    if not fix_seed:
        seed = random.randint(1, 2**31 - 1)
    
    control = {
        'map': map,
        'fim': fim,
        'll_is': ll_is,
        'll_gq': ll_gq,
        'nbiter_saemix': nbiter_saemix,
        'nbiter_sa': nbiter_sa,
        'nbiter_burn': nbiter_burn,
        'nbiter_map': nbiter_map,
        'nb_chains': nb_chains,
        'fix_seed': fix_seed,
        'seed': seed,
        'nmc_is': nmc_is,
        'nu_is': nu_is,
        'print_is': print_is,
        'nbdisplay': nbdisplay,
        'display_progress': display_progress,
        'print': print_results,
        'save': save,
        'save_graphs': save_graphs,
        'directory': directory,
        'warnings': warnings,
        'nbiter_mcmc': nbiter_mcmc,
        'proba_mcmc': proba_mcmc,
        'stepsize_rw': stepsize_rw,
        'rw_init': rw_init,
        'alpha_sa': alpha_sa,
        'nnodes_gq': nnodes_gq,
        'nsd_gq': nsd_gq,
        'maxim_maxiter': maxim_maxiter,
        'nb_sim': nb_sim,
        'nb_simpred': nb_simpred,
        'ipar_lmcmc': ipar_lmcmc,
        'ipar_rmcmc': ipar_rmcmc,
    }
    
    control.update(kwargs)
    control['nbiter_tot'] = sum(nbiter_saemix)
    
    return control
