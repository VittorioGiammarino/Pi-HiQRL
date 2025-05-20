from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, LogParam, LengthNormalize, Identity
from utils.networks import GCIQEValue, GCMRNValue, GCValue 

class PI_HIQRL_Agent(flax.struct.PyTreeNode):
    """Physics-informed Hierarchical Quasimetric RL (Pi-HiQRL) agent.

    This implementation supports the following variants:
    (1) High-level Value parameterizations: IQE (quasimetric_type='iqe') and MRN (quasimetric_type='mrn').
    (2) Low-level Value parameterized as an MLP and trained via IVL.
    (3) Actor: Hierarchical actor using AWR for both policies. 

    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)
    
    @jax.jit
    def high_distance_grad_s(self, obs, goals, grad_params):
        """Compute the quasimetric value function gradient penalty"""
        def distance_s(s, g, params):
            return self.network.select('high_value')(s, g, params=params).mean()
        
        grad_distance_s = jax.vmap(jax.grad(distance_s, argnums=0), in_axes=(0, 0, None))(obs, goals, grad_params)
        return grad_distance_s
    
    @jax.jit
    def low_distance_grad_s(self, obs, goals, grad_params):
        """Compute the value function gradient penalty"""
        def distance_s1(s, g, params):
            v1, _ = self.network.select('low_value')(s, g, params=params)
            d1 = -v1
            return d1.mean()
        
        def distance_s2(s, g, params):
            _, v2 = self.network.select('low_value')(s, g, params=params)
            d2 = -v2
            return d2.mean()
        
        grad_distance_s1 = jax.vmap(jax.grad(distance_s1, argnums=0), in_axes=(0, 0, None))(obs, goals, grad_params)
        grad_distance_s2 = jax.vmap(jax.grad(distance_s2, argnums=0), in_axes=(0, 0, None))(obs, goals, grad_params)

        return grad_distance_s1, grad_distance_s2

    def process_batch_high_value_loss(self, batch):

        if self.config['high_level_coordinates_only']:

            coord_num = self.config['coordinates_cardinality']

            if self.config['antmaze_soccer_flag']:
                batch_observations = jnp.concatenate([batch['observations'][:, 0:coord_num], batch['observations'][:, 15:17]], axis=-1) #adding coordinates ball
                batch_value_goals = jnp.concatenate([batch['value_goals'][:, 0:coord_num], batch['value_goals'][:, 15:17]], axis=-1)
                batch_next_observations = jnp.concatenate([batch['next_observations'][:, 0:coord_num], batch['next_observations'][:, 15:17]], axis=-1)

            else:
                batch_observations = batch['observations'][:, 0:coord_num]
                batch_value_goals = batch['value_goals'][:, 0:coord_num]
                batch_next_observations = batch['next_observations'][:, 0:coord_num]

        else:
            batch_observations = batch['observations']
            batch_value_goals = batch['value_goals']
            batch_next_observations = batch['next_observations']

        return batch_observations, batch_value_goals, batch_next_observations

    def high_value_loss(self, batch, grad_params):
        """Compute the QRL value loss."""

        batch_observations, batch_value_goals, batch_next_observations = self.process_batch_high_value_loss(batch)

        grad_distance_sg = self.high_distance_grad_s(batch_observations, batch_value_goals, grad_params)

        if self.config['projection']:
            s_dot = batch_next_observations - batch_observations
            grad_norm_d = jnp.sum(grad_distance_sg * s_dot, axis=-1)
        else:
            grad_norm_d = jnp.linalg.norm(grad_distance_sg + 1e-8, axis=-1) 

        eikonal_loss = (jax.nn.relu(grad_norm_d - 1) ** 2).mean() 

        total_loss = 0
        metrics = dict()
        
        d_sg = self.network.select('high_value')(batch_observations, batch_value_goals, params=grad_params)
        d_neg_loss = (100 * jax.nn.softplus(5 - d_sg / 100)).mean() # negative loss 

        total_loss += d_neg_loss 
        total_loss += eikonal_loss

        metrics['d_neg_loss'] = d_neg_loss
        metrics['Eikonal_loss'] = eikonal_loss

        metrics.update({'total_loss': total_loss,
            'd_sg_mean': d_sg.mean(),
            'd_sg_max': d_sg.max(),
            'd_sg_min': d_sg.min(),
            'grad_distance_sg_mean': grad_distance_sg.mean(),
            'grad_distance_sg_max': grad_distance_sg.max(),
            'grad_distance_sg_min': grad_distance_sg.min()
        })

        return total_loss, metrics
    
    def low_value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """

        grad_distance_sg1, grad_distance_sg2 = self.low_distance_grad_s(batch['observations'], batch['value_goals'], grad_params)

        if self.config['projection']:
            s_dot = batch['next_observations'] - batch['observations']
            grad_norm_d1 = jnp.sum(grad_distance_sg1 * s_dot, axis=-1)
            grad_norm_d2 = jnp.sum(grad_distance_sg2 * s_dot, axis=-1)

        else:
            grad_norm_d1 = jnp.linalg.norm(grad_distance_sg1 + 1e-8, axis=-1) 
            grad_norm_d2 = jnp.linalg.norm(grad_distance_sg2 + 1e-8, axis=-1) 

        eikonal_loss = jnp.mean((grad_norm_d1 - 1) ** 2) + jnp.mean((grad_norm_d2 - 1) ** 2) 

        (next_v1_t, next_v2_t) = self.network.select('target_low_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_low_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('low_value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2 + eikonal_loss

        return value_loss, {
            'value_loss': value_loss,
            'eikonal_loss': eikonal_loss
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss."""
        v = self.network.select('low_value')(batch['observations'], batch['low_actor_goals'])
        nv = self.network.select('low_value')(batch['next_observations'], batch['low_actor_goals'])
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Compute the goal representations of the subgoals.
        goal_reps = self.network.select('goal_rep')(jnp.concatenate([batch['observations'], 
                                                                     batch['low_actor_goals']], 
                                                                     axis=-1),
                                                                     params=grad_params)
        
        if not self.config['low_actor_rep_grad']:
            # Stop gradients through the goal representations.
            goal_reps = jax.lax.stop_gradient(goal_reps)

        dist = self.network.select('low_actor')(batch['observations'], 
                                                goal_reps, 
                                                goal_encoded=True, 
                                                params=grad_params)
        
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }

        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    def process_batch_high_actor_loss(self, batch):

        if self.config['high_level_coordinates_only']:
            coord_num = self.config['coordinates_cardinality']

            if self.config['antmaze_soccer_flag']:
                batch_observations = jnp.concatenate([batch['observations'][:, 0:coord_num], batch['observations'][:, 15:17]], axis=-1) #adding coordinates ball
                batch_high_actor_targets = jnp.concatenate([batch['high_actor_targets'][:, 0:coord_num], batch['high_actor_targets'][:, 15:17]], axis=-1)
                batch_high_actor_goals = jnp.concatenate([batch['high_actor_goals'][:, 0:coord_num], batch['high_actor_goals'][:, 15:17]], axis=-1)
            
            else:
                batch_observations = batch['observations'][:, 0:coord_num]
                batch_high_actor_targets = batch['high_actor_targets'][:, 0:coord_num]
                batch_high_actor_goals = batch['high_actor_goals'][:, 0:coord_num]

        else:
            batch_observations = batch['observations']
            batch_high_actor_targets = batch['high_actor_targets']
            batch_high_actor_goals = batch['high_actor_goals']

        return batch_observations, batch_high_actor_targets, batch_high_actor_goals
    
    def high_actor_loss(self, batch, grad_params):
        """Compute the high-level actor loss."""

        batch_observations, batch_high_actor_targets, batch_high_actor_goals = self.process_batch_high_actor_loss(batch)

        v = -self.network.select('high_value')(batch_observations, batch_high_actor_goals)
        nv = -self.network.select('high_value')(batch_high_actor_targets, batch_high_actor_goals)
        adv = nv - v 

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(batch_observations, 
                                                 batch_high_actor_goals, 
                                                 params=grad_params)
        
        target = self.network.select('goal_rep')(jnp.concatenate([batch['observations'], 
                                                                  batch['high_actor_targets']], 
                                                                  axis=-1))
        
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        low_value_loss, low_value_info = self.low_value_loss(batch, grad_params)
        for k, v in low_value_info.items():
            info[f'low_value/{k}'] = v

        high_value_loss, high_value_info = self.high_value_loss(batch, grad_params)
        for k, v in high_value_info.items():
            info[f'high_value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = low_value_loss + high_value_loss + low_actor_loss + high_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params
    
    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'low_value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor.

        It first queries the high-level actor to obtain subgoal representations, and then queries the low-level actor
        to obtain raw actions.
        """
        high_seed, low_seed = jax.random.split(seed)

        if self.config['high_level_coordinates_only']:
            coord_num = self.config['coordinates_cardinality']

            if self.config['antmaze_soccer_flag']:
                obs = jnp.concatenate([observations[:coord_num], observations[15:17]])
                goal = jnp.concatenate([goals[:coord_num], goals[15:17]])
                high_dist = self.network.select('high_actor')(obs, goal, temperature=temperature)

            else:
                high_dist = self.network.select('high_actor')(observations[:coord_num], goals[:coord_num], temperature=temperature)
                
        else:
            high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)

        goal_reps = high_dist.sample(seed = high_seed)

        # normalization
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        low_dist = self.network.select('low_actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations

        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define (state-dependent) subgoal representation phi([s; g]) that outputs a length-normalized vector.
        goal_rep_seq = []
        goal_rep_seq.append(MLP(hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                                activate_final=False,
                                layer_norm=config['layer_norm'],
                                ))
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        # Define encoders.
        # State-based environments only use the pre-defined shared encoder for subgoal representations.
        # Value: V(s, g)
        if config['encode_state']:
            encoder_seq = []
            encoder_seq.append(MLP(hidden_dims=(*config['value_hidden_dims'], config['encoder_latent_dim']),
                                activate_final=False,
                                layer_norm=config['layer_norm'],
                                ))
            encoder_def = nn.Sequential(encoder_seq)
            high_value_encoder_def = GCEncoder(state_encoder=encoder_def)
            
        else:
            high_value_encoder_def = GCEncoder(state_encoder=Identity())

        # Value: V(s, phi([s; g]))
        low_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
        target_low_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)

        # Low-level actor: pi^l(. | s, phi([s; w]))
        low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)

        # High-level actor: pi^h(. | s, g) (i.e., no encoder)
        high_actor_encoder_def = None

        # Define value and actor networks.
        if config['quasimetric_type'] == 'mrn':
            high_value_def = GCMRNValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                encoder=high_value_encoder_def,
            )

        elif config['quasimetric_type'] == 'iqe':
            high_value_def = GCIQEValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                dim_per_component=8,
                layer_norm=config['layer_norm'],
                encoder=high_value_encoder_def,
            )

        else:
            raise ValueError(f'Unsupported quasimetric type: {config["quasimetric_type"]}')
        
        low_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=low_value_encoder_def,
        )

        target_low_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_low_value_encoder_def,
        )

        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=low_actor_encoder_def,
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=low_actor_encoder_def,
            )

        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=high_actor_encoder_def,
        )

        if config['antmaze_soccer_flag']:
            coord_num = 4
        else:
            coord_num = config['coordinates_cardinality']

        if config['high_level_coordinates_only']:
            network_info = dict(
                goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
                high_value=(high_value_def, (ex_observations[:, 0:coord_num], ex_goals[:, 0:coord_num])),
                high_actor=(high_actor_def, (ex_observations[:, 0:coord_num], ex_goals[:, 0:coord_num])),
                low_value=(low_value_def, (ex_observations, ex_goals)),
                target_low_value=(target_low_value_def, (ex_observations, ex_goals)),
                low_actor=(low_actor_def, (ex_observations, ex_goals)),
            )

        else:
            network_info = dict(
                goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
                high_value=(high_value_def, (ex_observations, ex_goals)),
                high_actor=(high_actor_def, (ex_observations, ex_goals)),
                low_value=(low_value_def, (ex_observations, ex_goals)),
                target_low_value=(target_low_value_def, (ex_observations, ex_goals)),
                low_actor=(low_actor_def, (ex_observations, ex_goals)),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_low_value'] = params['modules_low_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='pi_hiqrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            quasimetric_type='iqe',  # Quasimetric parameterization type ('iqe' or 'mrn').
            latent_dim=512,  # Latent dimension for the quasimetric value function.
            encode_state = False, # encode input for the quasimetric value function
            encoder_latent_dim=2, #Latent dimension for the encoder processing input for the quasimetric value function
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor 
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=25,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            projection = False, # projecting gradient or not
            high_level_coordinates_only = False, 
            coordinates_cardinality = 2,
            antmaze_soccer_flag = False,
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
