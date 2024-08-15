# marquise 548
# eyrie 492
# alliance 530
# vagabond 604

from gymnasium import Space

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import logits_to_probs

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs

import config


ACTIONS = config.NUM_ACTIONS_EYRIE
OBS_LENGTH = config.NUM_OBS_EYRIE

RESNET_FEATURE_SIZE = 256
# final layer sizes before output layer
POLICY_FEATURE_SIZE = 128
VALUE_FEATURE_SIZE = 128

RESNET_DEPTH = 2
VALUE_DEPTH = 2
POLICY_DEPTH = 2

def split_input(obs, split):
    return   obs[:,:-split], obs[:,-split:]


class CustomDense(nn.Module):
    def __init__(self, input_dim, num_filters, batch_norm=False, activation='relu') -> None:
        super().__init__()
        self.dense = nn.Linear(in_features=input_dim, out_features=num_filters)
        self.batch_norm = nn.BatchNorm1d(num_filters, momentum=0.9) if batch_norm else None
        self.activation = getattr(F, activation) if activation else None
    
    def forward(self, x):
        x = self.dense(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)

        return x


class CustomResidual(nn.Module):
    def __init__(self, input_dim, num_filters) -> None:
        super().__init__()
        # this is just a placeholder linear layer, replaced later for specific input size
        self.dense1 = CustomDense(input_dim, num_filters)
        self.dense2 = CustomDense(num_filters, num_filters, activation=None)
        self.activation = F.relu
    
    def forward(self, x):
        shortcut = x

        x = self.dense1(x)
        x = self.dense2(x)
        x += shortcut
        x = self.activation(x)

        return x


class CustomResnetExtractor(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.seq = nn.Sequential(CustomDense(input_dim=input_dim, num_filters=RESNET_FEATURE_SIZE))
        for _ in range(RESNET_DEPTH):
            self.seq.append(CustomResidual(RESNET_FEATURE_SIZE, RESNET_FEATURE_SIZE))

    def forward(self, x):
        return self.seq(x)


class CustomValueHead(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.seq = nn.Sequential(CustomDense(input_dim=input_dim, num_filters=VALUE_FEATURE_SIZE*2))
        
        self.seq.append(CustomDense(input_dim=VALUE_FEATURE_SIZE*2, num_filters=VALUE_FEATURE_SIZE))
        
        self.seq.append(
            CustomDense(input_dim=VALUE_FEATURE_SIZE, num_filters=1, activation='tanh')
        )

    def forward(self, x):
        return self.seq(x)

# def value_head(y):
#     for _ in range(VALUE_DEPTH):
#         y = dense(y, FEATURE_SIZE)
#     vf = dense(y, 1, batch_norm = False, activation = 'tanh', name='vf')
#     q = dense(y, ACTIONS, batch_norm = False, activation = 'tanh', name='q')
#     return vf, q


class CustomPolicyHead(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.seq = nn.Sequential(CustomDense(input_dim=input_dim, num_filters=2*POLICY_FEATURE_SIZE))
        
        self.seq.append(CustomDense(input_dim=2*POLICY_FEATURE_SIZE, num_filters=POLICY_FEATURE_SIZE))
        
        self.seq.append(
            CustomDense(input_dim=POLICY_FEATURE_SIZE, num_filters=ACTIONS, activation=None)
        )

    def forward(self, x, legal_actions):
        mask = (1 - legal_actions) * -1e8

        return (self.seq(x) + mask)
    

# def policy_head(y, legal_actions):

#     for _ in range(POLICY_DEPTH):
#         y = dense(y, FEATURE_SIZE)
#     policy = dense(y, ACTIONS, batch_norm = False, activation = None, name='pi')
    
#     mask = Lambda(lambda x: (1 - x) * -1e8)(legal_actions)   
    
#     policy = Add()([policy, mask])
#     return policy


class FeatureMaskExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space, features_dim: int):
        super().__init__(observation_space, features_dim)
        self.extractor = CustomResnetExtractor(input_dim=OBS_LENGTH)
    
    def forward(self, x):
        obs, legal_actions = split_input(x, ACTIONS)
        features = self.extractor(obs)
        return features, legal_actions


class CustomMLPExtractor(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        last_layer_dim_pi: int = 8,
        last_layer_dim_vf: int = 8,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = CustomPolicyHead(input_dim=RESNET_FEATURE_SIZE)
        # Value network
        self.value_net = CustomValueHead(input_dim=RESNET_FEATURE_SIZE)

    def forward(self, features):
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features, legal_actions):
        return self.policy_net(features, legal_actions)

    def forward_critic(self, features):
        return self.value_net(features)


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            features_extractor_class=FeatureMaskExtractor,
            features_extractor_kwargs=dict(features_dim=RESNET_FEATURE_SIZE),
            **kwargs,
        )
        # self.device = get_device()

    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomMLPExtractor()

    def forward(self, obs, deterministic=False):
        features, legal_actions = self.features_extractor(obs)
        policy_logits = self.mlp_extractor.forward_actor(features, legal_actions)
        values = self.mlp_extractor.forward_critic(features)
        dist = CategoricalDistribution(ACTIONS)
        dist.proba_distribution(policy_logits)

        if deterministic:
            actions = torch.argmax(policy_logits, dim=1)
        else:
            actions = dist.sample()
        
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob
    
    def _predict(self, obs, deterministic=False):
        features, legal_actions = self.features_extractor(obs)
        policy_logits = self.mlp_extractor.forward_actor(features, legal_actions)
        dist = CategoricalDistribution(ACTIONS)
        dist.proba_distribution(policy_logits)

        if deterministic:
            actions = torch.argmax(policy_logits, dim=1)
        else:
            actions = dist.sample()
        
        return actions
    
    def evaluate_actions(self, obs, actions):
        features, legal_actions = self.features_extractor(obs)
        policy_logits = self.mlp_extractor.forward_actor(features, legal_actions)
        values = self.mlp_extractor.forward_critic(features)
        dist = CategoricalDistribution(ACTIONS)
        dist.proba_distribution(policy_logits)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features, legal_actions = self.features_extractor(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return latent_vf
    
    def action_probability(self, obs: PyTorchObs):
        "Return the action probabilities for each action according to the current policy given the observations."
        features, legal_actions = self.features_extractor(obs)
        policy_logits = self.mlp_extractor.forward_actor(features, legal_actions)
        return logits_to_probs(policy_logits)