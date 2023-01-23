import casadi as cs
from csnlp.util.math import quad_form
from sym_metanet import Network


def get_parametric_cost(
    network: Network,
) -> tuple[cs.Function, cs.Function, cs.Function]:
    """Creates parametric cost functions for the MPC controller. These are the
     - parametric initial cost: `V(s, weights)` (or \lambda_0)
     - paremtric stage cost: `L(s, rho_crit, v_free, weights)`
     - paremtric terminal cost: `T(s, rho_crit, v_free, weights)`

    Parameters
    ----------
    network : Network
        The network the MPC controller is built for.

    Returns
    -------
    tuple[cs.Function, cs.Function, cs.Function]
        A tuple of the 3 CasADi functions.
    """
    symvar = cs.MX  # faster function evaluations with MX

    # create variables
    n_segments = sum(link.N for _, _, link in network.links)
    n_origins = len(network.origins)
    rho = symvar.sym("rho", n_segments, 1)
    v = symvar.sym("v", n_segments, 1)
    w = symvar.sym("w", n_origins, 1)
    s = cs.vertcat(rho, v, w)
    rho_crit = symvar.sym("rho_crit", 1, 1)
    v_free = symvar.sym("v_free", 1, 1)

    # create initial cost function
    weight_init = symvar.sym("weight_init", n_segments * 2 + n_origins, 1)
    init_cost = cs.Function(
        "init_cost",
        (s, weight_init),
        (cs.dot(weight_init, s),),
        ("s", "weight"),
        ("J_init",),
    )

    # create stage cost function
    weight_stage1_rho = symvar.sym("weight_stage1_rho", n_segments, 1)
    weight_stage2_rho = symvar.sym("weight_stage2_rho", n_segments, 1)
    weight_stage_v = symvar.sym("weight_stage_v", n_segments, 1)
    stage_term1_rho = quad_form(weight_stage1_rho, rho - rho_crit)
    stage_term2_rho = quad_form(weight_stage2_rho, rho)
    stage_term_v = quad_form(weight_stage_v, v - v_free)
    weight_stage = cs.vertcat(weight_stage1_rho, weight_stage2_rho, weight_stage_v)
    stage_cost = cs.Function(
        "stage_cost",
        (s, rho_crit, v_free, weight_stage),
        (stage_term1_rho * stage_term2_rho + stage_term_v,),
        ("s", "rho_crit", "v_free", "weight"),
        ("J_stage",),
    )

    # create terminal cost function
    weight_fin1_rho = symvar.sym("weight_fin1_rho", n_segments, 1)
    weight_fin2_rho = symvar.sym("weight_fin2_rho", n_segments, 1)
    weight_fin_v = symvar.sym("weight_fin_v", n_segments, 1)
    fin_term1_rho = quad_form(weight_fin1_rho, rho - rho_crit)
    fin_term2_rho = quad_form(weight_fin2_rho, rho)
    fin_term_v = quad_form(weight_fin_v, v - v_free)
    weight_fin = cs.vertcat(weight_fin1_rho, weight_fin2_rho, weight_fin_v)
    fin_cost = cs.Function(
        "terminal_cost",
        (s, rho_crit, v_free, weight_fin),
        (fin_term1_rho * fin_term2_rho + fin_term_v,),
        ("s", "rho_crit", "v_free", "weight"),
        ("J_terminal",),
    )

    assert (
        init_cost.size_out(0) == (1, 1)
        and stage_cost.size_out(0) == (1, 1)
        and fin_cost.size_out(0) == (1, 1)
    ), "Invalid cost functions."
    return init_cost, stage_cost, fin_cost
