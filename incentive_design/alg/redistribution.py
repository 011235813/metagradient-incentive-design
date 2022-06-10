"""Tax computations based on foundation/components/redistribution.py

Functions here are applied onto the output of the ID's tax function, and
used to produce scalar modification to agents' rewards.
The ID can differentiate through these functions w.r.t. the parameters
of the tax function.

Based on foundation/components/redistribution.py
# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
import tensorflow as tf


class PeriodicBracketTax(object):

    def __init__(self,
                 n_agents=4,
                 cap_tax_rate=False,
                 rate_max=1.0,
                 rate_min=0.0,
                 usd_scaling=1000.0):

        self.n_agents = n_agents
        self.cap_tax_rate = cap_tax_rate

        # Assumes US federal
        self.bracket_cutoffs = (
            np.array([0, 9700, 39475, 84200, 160725, 204100, 510300])
            / 1000.0)
        self.n_brackets = len(self.bracket_cutoffs)
        self.bracket_edges = np.concatenate([self.bracket_cutoffs, [np.inf]])
        self.bracket_sizes = self.bracket_edges[1:] - self.bracket_edges[:-1]
        self.bracket_sizes = self.bracket_sizes.astype('float32')

    def income_bin(self, income):
        """Return index of tax bin in which income falls."""
        if income < 0:
            return 0.0
        meets_min = income >= self.bracket_edges[:-1]
        under_max = income < self.bracket_edges[1:]
        bracket_bool = meets_min * under_max
        return self.bracket_cutoffs[np.argmax(bracket_bool)]

    def marginal_rate(self, income):
        """Return the marginal tax rate applied at this income level."""
        if income < 0:
            return 0.0
        meets_min = income >= self.bracket_edges[:-1]
        under_max = income < self.bracket_edges[1:]
        bracket_bool = meets_min * under_max
        return self.curr_marginal_rates[np.argmax(bracket_bool)]

    def taxes_due(self, income, tax_rates):
        """
        Args:
        income: Placeholder [batch], agent's taxable income
        tax_rates: Tensor [batch, num_brackets], output of ID's tax function

        Returns Tensor [batch]
        """
        income = tf.reshape(income, [-1, 1])
        past_cutoff = tf.maximum(0.0, income - self.bracket_cutoffs)
        bin_income = tf.minimum(self.bracket_sizes, past_cutoff)
        bin_taxes = tax_rates * bin_income

        return tf.reduce_sum(bin_taxes, axis=1)

    def compute_tax_and_redistribution(self, income, tax_rates,
                                       curr_rate_max, agent_inventory_coin):
        """
        Args:
        income: Placeholder [time*n_agents], agent's taxable income
        tax_rates: Tensor [time, num_brackets], output of ID's tax function
        curr_rate_max: Tensor [time]
        agent_inventory_coin: Placeholder [time*n_agents]

        Based on enact_taxes() and taxes_due()
        """
        # Duplicate to get [time*n_agents, num_brackets]
        tax_rates = tf.reshape(tf.tile(tax_rates, [1, self.n_agents]),
                               [-1, self.n_brackets])

        # Duplicate to get [time*n_agents, 1]
        curr_rate_max = tf.reshape(tf.tile(
            tf.expand_dims(curr_rate_max, -1), [1, self.n_agents]), [-1, 1])

        if self.cap_tax_rate:
            # curr_marginal_rates = tf.minimum(tax_rates, curr_rate_max)
            curr_marginal_rates = tax_rates * curr_rate_max
        else:
            curr_marginal_rates = tax_rates

        # [time*n_agents]
        tax_due = self.taxes_due(income, curr_marginal_rates)

        # Don't take from escrow. [time*n_agents]
        effective_taxes = tf.minimum(agent_inventory_coin, tax_due)

        # Sum over agents for each time step
        # [time]
        net_tax_revenue = tf.reduce_sum(tf.reshape(
            effective_taxes, [-1, self.n_agents]), axis=1)
        # Duplicate to get [time*n_agents}
        net_tax_revenue = tf.reshape(tf.tile(
            tf.expand_dims(net_tax_revenue, -1), [1, self.n_agents]), [-1])

        lump_sum = net_tax_revenue / self.n_agents

        redistribution_minus_tax = lump_sum - effective_taxes

        return redistribution_minus_tax


class Utility(object):

    def __init__(self,
                 energy_cost=0.21,
                 energy_warmup_constant=0,
                 energy_warmup_method='decay',
                 isoelastic_eta=0.23):
        assert 0 <= isoelastic_eta <= 1.0
        self.energy_cost = energy_cost
        self.energy_warmup_constant = energy_warmup_constant
        self.energy_warmup_method = energy_warmup_method
        self.isoelastic_eta = isoelastic_eta

    def energy_weight(self, completions):
        """See foundation/scenarios/simple_wood_and_stone/layout_from_file.py

        Energy annealing progress. Multiply with self.energy_cost to get the
        effective energy coefficient.
        """
        if self.energy_warmup_constant <= 0.0:
            return 1.0

        if self.energy_warmup_method == "decay":
            return 1.0 - tf.math.exp(-completions / self.energy_warmup_constant)

        if self.energy_warmup_method == "auto":
            raise NotImplementedError
            # return float(
            #     1.0
            #     - np.exp(-self._auto_warmup_integrator / self.energy_warmup_constant)
            # )

        raise NotImplementedError

    def isoelastic_coin_minus_labor(self, coin_endowment, total_labor,
                                    completions):
        """See foundation/scenarios/utils/rewards.py

        Args:
            coin_endowment: TF tensor
            total_labor: TF tensor
            completionts: TF tensor

        Returns: TF tensor
        """
        labor_coefficient = self.energy_weight(completions) * self.energy_cost

        # https://en.wikipedia.org/wiki/Isoelastic_utility
        # assert np.all(coin_endowment >= 0)

        # Utility from coin endowment
        if self.isoelastic_eta == 1.0:  # dangerous
            # util_c = np.log(np.max(1, coin_endowment))
            util_c = tf.log(tf.math.maximum(1.0, coin_endowment) + 1e-15)
        else:  # isoelastic_eta >= 0
            # Prevent gradient from going to inf
            coin_endowment = tf.math.maximum(0.01, coin_endowment)
            util_c = ((coin_endowment**(1-self.isoelastic_eta)-1) /
                      (1 - self.isoelastic_eta))

        # disutility from labor
        util_l = total_labor * labor_coefficient

        # Net utility
        util = util_c - util_l

        return util        
