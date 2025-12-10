"""Comp Neuro Type Definitions

Defines custom type definitions to help with documentation and sphinx resolution
"""

import typing as tp
from scipy.integrate import OdeSolver

# Literal tag for the custom forward Euler solver implemented in BaseModel.solve
Euler = tp.Literal["Euler"]

# Accept Euler keyword, scipy's odeint string, any other solver name, or an OdeSolver instance
solvers = tp.Union[Euler, tp.Literal["scipy.integrate.odeint"], str, OdeSolver]
