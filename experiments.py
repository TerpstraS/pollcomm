#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 19/03/2022
# ---------------------------------------------------------------------------
""" experiments.py

Functions to perform the experiments. These functions are called from ms.py
"""
# ---------------------------------------------------------------------------
import copy
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from tqdm import tqdm

import pollcomm as pc

from cycler import cycler
line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))

plt.rc("axes", prop_cycle=line_cycler)
# plt.rc("axes", prop_cycle=marker_cycler)
plt.rc("font", family="serif", size=18.)
plt.rc("savefig", dpi=200)
plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)

MODELS = {
    "BM": pc.BaseModel,
    "AM": pc.AdaptiveModel
}


def state_space_rate_dA(fname, AM, rates=None, dAs_init=None, A_init=None):

    if rates is None:
        rates = np.linspace(0.0001, 0.1, 11)
    if dAs_init is None:
        dAs_init = np.linspace(0, 4, 11)
    if A_init is None:
        A_init = 1

    def dA_rate(t, r, dA_init):
        return dA_init + r * t

    t_end = int(1e5)
    n_steps = int(1e6) # number of interpolated time steps
    extinct_threshold = 0.01

    dAs_critical = np.zeros((len(rates), len(dAs_init)))

    curr_iter = 0
    total_iter = len(rates) * len(dAs_init)

    for i, rate in enumerate(rates):
        for j, dA_init in enumerate(dAs_init):
            print(f"Iteration {curr_iter + 1} out of {total_iter}")

            # initial conditions
            y0 = np.full(AM.N, A_init, dtype=float)
            y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))

            # drivers of decline
            dA = {
                "func": dA_rate,
                "args": (rate, dA_init)
            }

            sol = AM.solve(
                t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=True,
                extinct_threshold=extinct_threshold
            )

            # check if point of collapse has been found:
            if sol.status == 1:

                # find point of collapse
                A = AM.y[AM.N_p:AM.N]

                # put default at -1 if no population went extinct
                try:
                    ind = (A < extinct_threshold).all(axis=0).nonzero()[0][0]
                    t_extinct = AM.t[ind]
                    dAs_critical[i, j] = dA["func"](t_extinct, rate, dA_init)
                except IndexError:
                    dAs_critical[i, j] = -1
            else:
                dAs_critical[i, j] = -1

            curr_iter += 1

        np.savez(
            fname, rates=rates, dAs_init=dAs_init, A_init=A_init,
            dAs_critical=dAs_critical,
        )


def state_space_abundance_dA(fname, AM, dAs_init=None, A_init=None):

    if dAs_init is None:
        dAs_init = np.linspace(0, 4, 41)
    if A_init is None:
        A_init = np.linspace(0, 1, 11)

    t_end = int(1e5)
    n_steps = int(1e4) # number of interpolated time steps

    final_abundance = np.zeros((len(dAs_init), len(A_init)))

    curr_iter = 0
    total_iter = len(dAs_init) * len(A_init)

    for i, dA in enumerate(dAs_init):
        for j, abundance in enumerate(A_init):
            print(f"Iteration {curr_iter + 1} out of {total_iter}")

            # initial conditions
            y0 = np.full(AM.N, abundance, dtype=float)
            y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))

            sol = AM.solve(
                t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=False,
                stop_on_equilibrium=True
            )

            A_mean = AM.y[AM.N_p:AM.N].mean(axis=0)

            # final abundace is the mean abundace at the final time point
            final_abundance[i, j] = A_mean[-1]

            curr_iter += 1

        np.savez(
            fname, dAs_init=dAs_init, A_init=A_init, final_abundance=final_abundance,
        )


def state_space_abundance_rate_critical_dA(fname, AM, rates=None, A_init=None):

    if rates is None:
        rates = np.linspace(0.0001, 0.1, 11)
    if A_init is None:
        A_init = np.linspace(0, 1, 11)

    def dA_rate(t, r):
        return r * t

    t_end = int(1e5)
    n_steps = int(1e4) # number of interpolated time steps
    extinct_threshold = 0.01

    dAs_critical = np.zeros((len(rates), len(A_init)))

    curr_iter = 0
    total_iter = len(rates) * len(A_init)

    for i, rate in enumerate(rates):
        for j, abundance in enumerate(A_init):
            print(f"Iteration {curr_iter + 1} out of {total_iter}")

            # initial conditions
            y0 = np.full(AM.N, abundance, dtype=float)
            y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))

            # drivers of decline
            dA = {
                "func": dA_rate,
                "args": (rate, )
            }

            sol = AM.solve(
                t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=True,
                extinct_threshold=extinct_threshold
            )

            # check if point of collapse has been found:
            if sol.status == 1:

                # find point of collapse
                A_mean = AM.y[AM.N_p:AM.N].mean(axis=0)

                # put default at -1 if no population went extinct
                try:
                    ind = (A_mean < extinct_threshold).nonzero()[0][0]
                    t_extinct = AM.t[ind]
                    dAs_critical[i, j] = dA["func"](t_extinct, rate)
                except IndexError:
                    dAs_critical[i, j] = -1
            else:
                dAs_critical[i, j] = -1

            curr_iter += 1

        np.savez(
            fname, rates=rates, A_init=A_init, dAs_critical=dAs_critical,
        )


def state_space_abundance_rate_critical_dA_all(fname, AM, rates=None, A_init=None):

    if rates is None:
        rates = np.linspace(0.0001, 0.1, 11)
    if A_init is None:
        A_init = np.linspace(0, 1, 11)

    def dA_rate(t, r):
        return r * t

    t_end = int(1e5)
    n_steps = int(1e6) # number of interpolated time steps
    extinct_threshold = 0.01

    dAs_critical = np.zeros((len(rates), len(A_init)))

    curr_iter = 0
    total_iter = len(rates) * len(A_init)

    for i, rate in enumerate(rates):
        for j, abundance in enumerate(A_init):
            print(f"Iteration {curr_iter + 1} out of {total_iter}")

            # initial conditions
            y0 = np.full(AM.N, abundance, dtype=float)
            y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))

            # drivers of decline
            dA = {
                "func": dA_rate,
                "args": (rate, )
            }

            sol = AM.solve(
                t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=True,
                extinct_threshold=extinct_threshold
            )

            # check if point of collapse has been found:
            if sol.status == 1:

                # find point of collapse
                A = AM.y[AM.N_p:AM.N]

                # print((A[:, -1] < extinct_threshold).all())
                # print((A < extinct_threshold).all(axis=0))
                # if not (A[:, -1] < extinct_threshold).all():
                #     print(A[:, -1])
                #     print(AM.t[-1])
                # put default at -1 if no population went extinct
                try:
                    ind = (A < extinct_threshold).all(axis=0).nonzero()[0][0]
                    t_extinct = AM.t[ind]
                    dAs_critical[i, j] = dA["func"](t_extinct, rate)
                except IndexError:
                    dAs_critical[i, j] = -1
            else:
                dAs_critical[i, j] = -1

            curr_iter += 1

        np.savez(
            fname, rates=rates, A_init=A_init, dAs_critical=dAs_critical,
        )


def state_space_rate_critical_dA(fname, AM, rates=None, A_init=None):

    if rates is None:
        rates = np.linspace(0.0001, 0.1, 11)
    if A_init is None:
        A_init = [0.2]

    def dA_rate(t, r):
        return r * t

    t_end = int(1e5)
    n_steps = int(1e6) # number of interpolated time steps
    extinct_threshold = 0.01

    dAs_critical = np.zeros((len(rates), len(A_init)))
    curr_iter = 0
    total_iter = len(rates) * len(A_init)

    for i, rate in enumerate(rates):
        for j, abundance in enumerate(A_init):
            print(f"Iteration {curr_iter + 1} out of {total_iter}")

            # initial conditions
            y0 = np.full(AM.N, abundance, dtype=float)
            y0 = np.concatenate((y0, copy.deepcopy(AM.alpha.flatten())))

            # drivers of decline
            dA = {
                "func": dA_rate,
                "args": (rate, )
            }

            sol = AM.solve(
                t_end, dA=dA, n_steps=n_steps, y0=y0, stop_on_collapse=True,
                extinct_threshold=extinct_threshold
            )

            # check if point of collapse has been found:
            if sol.status == 1:

                # find point of collapse
                A = AM.y[AM.N_p:AM.N]

                # print((A[:, -1] < extinct_threshold).all())
                # print((A < extinct_threshold).all(axis=0))
                # if not (A[:, -1] < extinct_threshold).all():
                #     print(A[:, -1])
                #     print(AM.t[-1])
                # put default at -1 if no population went extinct
                try:
                    ind = (A < extinct_threshold).all(axis=0).nonzero()[0][0]
                    t_extinct = AM.t[ind]
                    dAs_critical[i, j] = dA["func"](t_extinct, rate)
                except IndexError:
                    dAs_critical[i, j] = -1
            else:
                dAs_critical[i, j] = -1

            curr_iter += 1
    np.savez(
        fname, rates=rates, A_init=A_init, dAs_critical=dAs_critical,
    )


def hysteresis_q(AM, dAs=None, qs=None, seed=None, fnumber=0):
    """Calculate hysteresis as function of dA for different q. """
    if dAs is None:
        dAs = np.linspace(0, 4, 21)
    if qs is None:
        qs = np.linspace(0, 1, 11)
    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    rng = np.random.default_rng(seed)

    for i, q in enumerate(qs):
        print(f"\nCalculating q: {i+1} out of {len(qs)}...")

        # set q parameter, set seed and rng
        AM.q = q
        AM.rng = rng

        fname = f"output/hysteresis_G{AM.G}_nu{AM.nu}_q{AM.q}_{fnumber}"

        hysteresis(fname, AM, dAs=dAs)


def find_dA_collapse_recovery(fname, AM, dA_step=0.02):
    """Calculate hysteresis as function of dA. """
    # maximum simulation time
    t_end = int(1e4)
    t_step = int(1e4)
    extinct_threshold = 0.01

    # obtain initial solution
    y0 = AM.equilibrium()

    if AM.is_all_alive()[-1]:
        is_feasible = True
    else:
        is_feasible = False

    # calculate solution for increasing dA
    print("\nCalculating hysteresis forward...")
    dA = 0
    while (AM.y[AM.N_p:AM.N, -1] > extinct_threshold).any():

        dA += dA_step
        AM.solve(t_step, dA=dA, y0=y0, save_period=0, stop_on_equilibrium=True)


        y0 = AM.y[:, -1]
        y0 = np.concatenate((y0, AM.y_partial[:, -1]))

    dA_collapse = dA

    # calculate solution for decreasing dA
    print("\nCalculating hysteresis backward...")
    while not (AM.y[AM.N_p:AM.N, -1] > extinct_threshold).any():

        dA -= dA_step
        AM.solve(t_step, dA=dA, y0=y0, save_period=0, stop_on_equilibrium=True)

        y0 = AM.y[:, -1]
        y0 = np.concatenate((y0, AM.y_partial[:, -1]))

    dA_recover = dA

    np.savez(
        fname, dA_collapse=dA_collapse, dA_recover=dA_recover, is_feasible=is_feasible
    )


def hysteresis(fname, AM, dAs=None):
    """Calculate hysteresis as function of dA. """
    # maximum simulation time
    t_end = 1000
    t_step = 100

    if dAs is None:
        dAs = np.linspace(0, 4, 21)

    # save only steady state solutions
    P_sol_forward = np.zeros((len(dAs), AM.N_p))
    A_sol_forward = np.zeros((len(dAs), AM.N_a))
    P_sol_backward = np.zeros((len(dAs), AM.N_p))
    A_sol_backward = np.zeros((len(dAs), AM.N_a))

    # obtain initial solution
    AM.solve(t_end, dA=0, save_period=0, stop_on_equilibrium=True)

    if AM.is_all_alive()[-1]:
        is_feasible = True
    else:
        is_feasible = False

    # calculate solution for increasing dA
    print("\nCalculating hysteresis forward...")
    for i, dA in enumerate(dAs):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = AM.y[:, -1]
        y0 = np.concatenate((y0, AM.y_partial[:, -1]))

        AM.solve(t_step, dA=dA, y0=y0, save_period=0, stop_on_equilibrium=True)

        P_sol_forward[i] = AM.y[:AM.N_p, -1]
        A_sol_forward[i] = AM.y[AM.N_p:AM.N, -1]

    # calculate solution for decreasing dA
    print("\nCalculating hysteresis backward...")
    for i, dA in enumerate(np.flip(dAs)):
        print(f"{i+1} out of {len(dAs)}...")

        y0 = AM.y[:, -1]
        y0 = np.concatenate((y0, AM.y_partial[:, -1]))

        AM.solve(t_step, dA=dA, y0=y0, save_period=0, stop_on_equilibrium=True)

        P_sol_backward[i] = AM.y[:AM.N_p, -1]
        A_sol_backward[i] = AM.y[AM.N_p:AM.N, -1]

    np.savez(
        fname, dAs=dAs, P_sol_forward=P_sol_forward, A_sol_forward=A_sol_forward,
        P_sol_backward=P_sol_backward, A_sol_backward=A_sol_backward,
        is_feasible=is_feasible
    )


def hysteresis_rate(fname, AM, rate=0.001, dA_max=3):

    # maximum simulation time
    t_equilibrium = 1000

    # time needed to reach dA_max for given rate
    t_end = dA_max / rate
    n_steps = 1000

    # save only steady state solutions
    P_sol_forward = np.zeros((n_steps, AM.N_p))
    A_sol_forward = np.zeros((n_steps, AM.N_a))
    P_sol_backward = np.zeros((n_steps, AM.N_p))
    A_sol_backward = np.zeros((n_steps, AM.N_a))

    if AM.is_all_alive()[-1]:
        is_feasible = True
    else:
        is_feasible = False

    # obtain initial solution
    AM.solve(
        t_equilibrium, n_steps=1000, dA=0, save_period=0, stop_on_equilibrium=True
    )

    # calculate solution for increasing dA
    print("\nCalculating hysteresis forward...")
    dA_rate = {
        "func": lambda t, rate: rate * t,
        "args": (rate, )
    }
    y0 = AM.y[:, -1]
    y0 = np.concatenate((y0, AM.y_partial[:, -1]))
    AM.solve(
        t_end, y0=y0, n_steps=n_steps, dA=dA_rate, save_period=0
    )
    P_sol_forward = AM.y[:AM.N_p].T
    A_sol_forward = AM.y[AM.N_p:AM.N].T

    dAs_forward = rate * AM.t

    # calculate solution for decreasing dA
    print("\nCalculating hysteresis backward...\n")
    def func(t, rate, dA_max):
        if dA_max - rate * t < 0:
            return 0
        else:
            return dA_max - rate * t
    dA_rate = {
        "func": func,
        "args": (rate, dA_max)
    }
    y0 = AM.y[:, -1]
    y0 = np.concatenate((y0, AM.y_partial[:, -1]))

    # make sure system ends up in final equilibrium
    AM.solve(
        t_end*10, y0=y0, n_steps=n_steps, dA=dA_rate, save_period=0,
        stop_on_equilibrium=True
    )
    P_sol_backward = AM.y[:AM.N_p].T
    A_sol_backward = AM.y[AM.N_p:AM.N].T

    dAs_backward = [func(t, rate, dA_max) for t in AM.t]

    np.savez(
        fname, dAs_forward=dAs_forward, dAs_backward=dAs_backward,
        P_sol_forward=P_sol_forward, A_sol_forward=A_sol_forward,
        P_sol_backward=P_sol_backward, A_sol_backward=A_sol_backward,
        is_feasible=is_feasible, rate=rate
    )
