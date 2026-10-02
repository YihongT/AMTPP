"""Independent analytic checks of the asymmetric log-Laplace distribution."""
import math
import unittest

import torch

from amtpp.models.amtpp import al_component_logpdf, almixture_log_prob_tau, almixture_quantile_tau


class DensityTest(unittest.TestCase):
    def test_density_integrates_to_one_and_has_hours_jacobian(self):
        beta, rate, asymmetry = 0.3, 1.7, 0.6
        y = torch.linspace(-40, 40, 40001, dtype=torch.float64)
        pdf = al_component_logpdf(y, torch.tensor(beta), torch.tensor(rate), torch.tensor(asymmetry)).exp()
        self.assertAlmostEqual(torch.trapezoid(pdf, y).item(), 1.0, places=5)
        tau = torch.tensor([[0.2, 1.0, 4.0]], dtype=torch.float64)
        shape = (*tau.shape, 1)
        actual = almixture_log_prob_tau(tau, torch.ones(shape, dtype=tau.dtype), torch.full(shape, beta), torch.full(shape, rate), torch.full(shape, asymmetry))
        a, b = rate / asymmetry, rate * asymmetry
        expected = [math.log(a * b / (a + b)) + (a * (math.log(t) - beta) if math.log(t) < beta else -b * (math.log(t) - beta)) - math.log(t) for t in tau[0].tolist()]
        torch.testing.assert_close(actual[0], torch.tensor(expected, dtype=tau.dtype), rtol=1e-6, atol=1e-6)

    def test_quantiles_match_analytic_inverse_cdf(self):
        beta, rate, asymmetry = 0.3, 1.7, 0.6
        a, b = rate / asymmetry, rate * asymmetry
        left_mass = b / (a + b)
        parameters = [torch.tensor([[[value]]], dtype=torch.float64) for value in (1.0, beta, rate, asymmetry)]
        for quantile in (0.1, 0.5, 0.9):
            log_time = beta + math.log(quantile / left_mass) / a if quantile < left_mass else beta - math.log((1 - quantile) / (1 - left_mass)) / b
            actual = almixture_quantile_tau(*parameters, quantile=quantile).item()
            self.assertAlmostEqual(actual, math.exp(log_time), places=9)
