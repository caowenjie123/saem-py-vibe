from typing import Optional, cast

import numpy as np
import pandas as pd

from saemix.algorithm.initialization import initialise_main_algo
from saemix.algorithm.saem import run_saem
from saemix.control import saemix_control
from saemix.data import SaemixData
from saemix.model import SaemixModel
from saemix.results import SaemixObject
from saemix.utils import cutoff, transphi


def saemix(
    model: SaemixModel, data: SaemixData, control: Optional[dict] = None
) -> SaemixObject:
    if control is None:
        control = saemix_control()

    if not isinstance(model, SaemixModel):
        raise TypeError("model must be a SaemixModel instance")
    if not isinstance(data, SaemixData):
        raise TypeError("data must be a SaemixData instance")

    yobs_transformed = None
    if isinstance(model.error_model, list) and "exponential" in model.error_model:
        data_frame = cast(pd.DataFrame, data.data)
        yname = data.name_response
        ytype = (
            data_frame["ytype"].to_numpy() if "ytype" in data_frame.columns else None
        )
        ytype_arr = np.asarray(ytype, dtype=int) if ytype is not None else None
        if ytype_arr is not None and len(model.error_model) > 1:
            if ytype_arr.min() >= 1 and ytype_arr.max() == len(model.error_model):
                ytype_arr = ytype_arr - 1
        if data.yorig is None:
            data.yorig = data_frame[yname].copy()
        yvals = data_frame[yname].to_numpy().copy()
        for ityp, em in enumerate(model.error_model):
            if em != "exponential":
                continue
            if ytype_arr is None:
                mask = np.ones(len(yvals), dtype=bool)
            else:
                mask = ytype_arr == ityp
            if np.any(mask):
                yvals[mask] = np.log(cutoff(yvals[mask]))
        yobs_transformed = yvals

    if yobs_transformed is not None:
        control["yobs_transformed"] = yobs_transformed

    saemix_object = SaemixObject(data=data, model=model, options=control)

    xinit = initialise_main_algo(data, model, control)
    model = xinit["saemix_model"]
    Dargs = xinit["Dargs"]
    Uargs = xinit["Uargs"]
    varList = xinit["varList"]
    opt = xinit["opt"]
    mean_phi = xinit["mean_phi"]
    phiM = xinit.get("phiM", None)

    structural_model = model.model

    results = run_saem(
        Dargs,
        Uargs,
        varList,
        opt,
        mean_phi,
        phiM,
        structural_model,
        rng=control.get("rng"),
    )

    saemix_object.results.mean_phi = results["mean_phi"]
    saemix_object.results.omega = results["varList"]["omega"]
    saemix_object.results.respar = results["varList"].get("pres", model.error_init)
    saemix_object.results.npar_est = Uargs.get("nb_parest")
    saemix_object.results.cond_mean_phi = results.get("cond_mean_phi", None)
    saemix_object.results.cond_var_phi = results.get("cond_var_phi", None)
    saemix_object.results.cond_mean_psi = results.get("cond_mean_psi", None)
    saemix_object.results.cond_mean_eta = results.get("cond_mean_eta", None)

    # Store iteration history
    saemix_object.results.parpop = results.get("parpop", None)
    saemix_object.results.allpar = results.get("allpar", None)

    mean_psi = transphi(results["mean_phi"], model.transform_par)
    if saemix_object.results.cond_mean_psi is None:
        saemix_object.results.cond_mean_psi = mean_psi
    if saemix_object.results.cond_mean_phi is None:
        saemix_object.results.cond_mean_phi = results["mean_phi"]

    if control.get("map", True):
        try:
            from saemix.algorithm.map_estimation import map_saemix

            saemix_object = map_saemix(saemix_object)
        except Exception as e:
            if control.get("warnings", True):
                print(f"Problem estimating the MAP parameters: {e}")

    if control.get("fim", True):
        try:
            from saemix.algorithm.fim import fim_saemix

            saemix_object = fim_saemix(saemix_object)
        except Exception as e:
            if control.get("warnings", True):
                print(f"Problem estimating the FIM: {e}")

    if control.get("ll_is", False):
        try:
            from saemix.algorithm.likelihood import llis_saemix

            saemix_object = llis_saemix(saemix_object)
        except Exception as e:
            if control.get("warnings", True):
                print(f"Problem estimating the log-likelihood (IS): {e}")

    if control.get("ll_gq", False):
        try:
            from saemix.algorithm.likelihood import llgq_saemix

            saemix_object = llgq_saemix(saemix_object)
        except Exception as e:
            if control.get("warnings", True):
                print(f"Problem estimating the log-likelihood (GQ): {e}")

    return saemix_object
