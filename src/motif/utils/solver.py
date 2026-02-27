"""Implements classes to solve the flow matching ODE for multi-sources data."""

from typing import Callable, Dict

import torch

from motif.datatypes import (
    MultisourceTensor,
    SourceIndex,
)


class MultisourceEulerODESolver:
    """Solves the ODE (df/dt)(x, t) = u(x, t) with the Euler method."""

    def __init__(self, vf_func: Callable[[MultisourceTensor, torch.Tensor], MultisourceTensor]):
        """
        Args:
            vf_func (Callable): Function that computes the vector field u(x, t)
                for each source. If a source is not present in the returned dict, it is assumed
                that the vector field for that source is zero.
        """
        self.vf_func = vf_func

    def solve(
        self, x_0: MultisourceTensor, time_grid: torch.Tensor, return_intermediate_steps=True
    ) -> MultisourceTensor | Dict[SourceIndex, torch.Tensor]:
        """Solves the ODE for the given initial conditions.
        Args:
            x_0 (MultisourceTensor): Initial conditions, dict {source: x_s} where x_s is a tensor
                of shape (B, C, ...) giving the initial values of the flow for the
                source s.
            time_grid (torch.Tensor): Time grid of shape (T,) within [0, 1]; times at which
                the flow will be evaluated. Also defines the step size for the Euler
                method.
            return_intermediate_steps (bool): If True, returns the intermediate solutions at
                each time step.
        Returns:
            If return_intermediate_steps is True, returns a dict {source: sol_s} where sol_s is
                a tensor of shape (T, B, C, ...) giving the solution of the flow for the
                source s at each time step. If False, returns a dict {source: final_sol_s}
                where final_sol_s is a tensor of shape (B, C, ...) giving the solution at
                the final time step.
        """
        device = next(iter(x_0.values())).device
        time_grid = time_grid.to(device)
        x = {source: x_0[source].to(device) for source in x_0}

        sol = {}
        # For each source, the returned solution will be a tensor of shape (T, B, C, ...)
        for source in x:
            sol[source] = torch.empty((len(time_grid), *x[source].shape), device=device)
            sol[source][0] = x[source]

        for k, (t0, t1) in enumerate(zip(time_grid[:-1], time_grid[1:])):
            u = self.vf_func(x, t0)
            for source in x:
                dt = t1 - t0
                if source in u:
                    x[source] = x[source] + dt * u[source]
                sol[source][k + 1] = x[source]

        if not return_intermediate_steps:
            sol = {source: sol[source][-1] for source in sol}
        return sol
