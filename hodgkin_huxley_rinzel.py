# pylint:disable=C0103

from collections import OrderedDict
import numpy as np
from ..base_model import BaseModel


class HodgkinHuxleyRinzel(BaseModel):
    """Rinzel Neuron Model

    The Rinzel Neuron Model is a reduced version of the Hodgkin-Huxley Neuron
    model. Instead of having 4 state variables :math:`(V, n, m, h)`, rinzel
    only has 2 state variables :math:`(V, W)`.

    .. note::

        In some texts, the state variables :math:`W` may also be called
        :math:`R`.
    """

    Time_Scale = 1e3  # s to ms
    """Rinzel Operates on Millisecond Scale"""

    Default_States = OrderedDict(V=-65.0, R=(0.0, 0.0, 1.0))
    """Default State Variables of the Rinzel Model"""

    Default_Params = OrderedDict(
        C=1.0,
        g_Na=120.0,
        g_K=36.0,
        g_L=0.3,
        E_Na=50.0,
        E_K=-77.0,
        E_L=-54.387,
        S=1.27135220916422,
    )
    """Default Parameters of the Rinzel Model"""

    def ode(self, t, states, I_ext=0.0):
        """Rinzel gradient function"""
        V, R = states

        alpha_n = (
            0.01 * (10.0 - (V + 65.0)) / (np.exp((10.0 - (V + 65.0)) / 10.0) - 1.0)
        )
        alpha_m = 0.1 * (25.0 - (V + 65.0)) / (np.exp((25.0 - (V + 65.0)) / 10.0) - 1.0)
        alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)

        beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
        beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
        beta_h = 1.0 / (np.exp((30.0 - (V + 65.0)) / 10.0) + 1.0)

        n_infty = alpha_n / (alpha_n + beta_n)
        m_infty = alpha_m / (alpha_m + beta_m)
        h_infty = alpha_h / (alpha_h + beta_h)
        R_infty = (
            self.params["S"]
            / (1 + self.params["S"] ** 2)
            * (n_infty + self.params["S"] * (1 - h_infty))
        )

        tau_R = 1.0 + 5.0 * np.exp(-(((V + 65.0) - 10.0) ** 2) / 55.0 ** 2)

        d_R = 3.0 * R_infty / tau_R - 3.0 * R / tau_R

        # Update the ionic currents and membrane voltage:
        I_K = (
            self.params["g_K"] * (R / self.params["S"]) ** 4 * (V - self.params["E_K"])
        )
        I_Na = self.params["g_Na"] * m_infty ** 3 * (1 - R) * (V - self.params["E_Na"])
        I_L = self.params["g_L"] * (V - self.params["E_L"])
        d_V = (I_ext - I_K - I_Na - I_L) / self.params["C"]

        return [d_V, d_R]
