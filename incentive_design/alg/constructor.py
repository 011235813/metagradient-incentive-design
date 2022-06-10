from incentive_design.alg import actor_critic
from incentive_design.alg import actor_critic_ps
from incentive_design.alg import ppo_agent


def get_designer(config, env, phase):
    """Creates a designer object."""

    if phase == 1 or config.alg.name in ['us_federal', 'free_market']:
        designer = None
    elif phase == 2 and config.alg.name =='dual_RL':
        # ID is an RL agent in dual_RL
        if config.alg.update_alg == 'ppo':
            designer = ppo_agent.Agent(
                'designer', config.designer, env.p_action_spaces,
                env.obs_dim_p_flat, env.obs_dim_p_t, 1, config.alg.objective,
                config.nn_designer)
        elif config.alg.update_alg == 'ac':
            designer = actor_critic.ActorCritic(
                'designer', config.designer, env.p_action_spaces,
                env.obs_dim_p_flat, env.obs_dim_p_t, config.nn_designer)
        else:
            raise ValueError('%s is not an option' % update_alg)
    elif phase == 2 and config.alg.name in ['m1', 'm2']:
        # AC and PPO versions of m1 and m2 both use id_foundation
        from incentive_design.alg import id_foundation
        designer = id_foundation.MetaGrad1Step(
            0, 'designer', config.designer_m,
            env.periodic_bracket_tax.n_brackets,
            env.obs_dim_p_flat, env.obs_dim_p_t, env.n_agents,
            config.nn_designer)
    else:
        raise ValueError('%s is not an option' % config.alg.name)

    return designer


def get_agents(config, env):
    """Creates agent object(s)."""

    if config.alg.update_alg == 'ppo':
        if config.alg.name in ['m1', 'm2']:
            if config.alg.name == 'm1':
                from incentive_design.alg import ppo_agent_m1 as ppom
            else:
                from incentive_design.alg import ppo_agent_m2 as ppom
            agents = ppom.Agent(
                1, 'agent', config.agent, env.agent_action_space,
                env.obs_dim_agent_flat, env.obs_dim_agent_t,
                env.n_agents, config.alg.objective, config.nn_agent,
                None, None) # tax and utility are not needed for test
        elif config.agent.share_parameter:
            agents = ppo_agent.Agent(
                'agent', config.agent, env.agent_action_space,
                env.obs_dim_agent_flat, env.obs_dim_agent_t,
                env.n_agents, config.alg.objective, config.nn_agent)
        else:
            agents = []
            for agent_id in range(0, env.n_agents):
                agents.append(ppo_agent.Agent(
                    'agent_%d'%agent_id, config.agent, env.agent_action_space,
                    env.obs_dim_agent_flat, env.obs_dim_agent_t,
                    1, config.alg.objective, config.nn_agent))
    else:
        if config.agent.share_parameter:
            if config.alg.name in ['m1', 'm2']:
                if config.alg.name == 'm1':
                    from incentive_design.alg import actor_critic_ps_m1 as acm
                else:
                    from incentive_design.alg import actor_critic_ps_m2 as acm
                agents = acm.ActorCritic(
                    1, 'agent', config.agent, env.agent_action_space[0],
                    env.obs_dim_agent_flat, env.obs_dim_agent_t,
                    env.n_agents, config.nn_agent, None, None)
            else:
                agents = actor_critic_ps.ActorCritic(
                    'agent', config.agent, env.agent_action_space[0],
                    env.obs_dim_agent_flat, env.obs_dim_agent_t,
                    env.n_agents, config.nn_agent)
        else:
            agents = []
            for agent_id in range(0, env.n_agents):
                agents.append(
                    actor_critic.ActorCritic(
                        'agent_%d'%agent_id, config.agent,
                        env.agent_action_space, env.obs_dim_agent_flat,
                        env.obs_dim_agent_t, config.nn_agent))

    return agents


    
