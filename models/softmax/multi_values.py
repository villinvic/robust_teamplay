from typing import List

import numpy as np
from gymnasium.spaces import Dict, Discrete
from ray.rllib import SampleBatch
from ray.rllib.models import ModelV2
from ray.rllib.models.tf import TFModelV2
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import override, try_import_tf
from ray.rllib.utils.typing import TensorType

tf1, tf, tfv = try_import_tf()

class MultiValueSoftmax(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **custom_args):
        model_config.update(**custom_args)

        self.num_outputs = action_space.n
        self.n_scenarios = model_config["n_scenarios"]

        super().__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )
        self.view_requirements[SampleBatch.INFOS] = ViewRequirement(
            SampleBatch.INFOS, shift=0, space=Dict(
                {
                    "scenario": Discrete(self.n_scenarios)
                }
            ), used_for_training=True, used_for_compute_actions=True
        )

        obs_input = tf.keras.layers.Input(shape=obs_space.shape, name="obs_input", dtype=tf.float32)
        scenario_mask = tf.keras.layers.Input(shape=(self.n_scenarios,), name="scenario_mask", dtype=tf.float32)

        action_logits = tf.keras.layers.Dense(
            self.num_outputs,
            name="action_logits",
            activation="linear"
        )(obs_input)

        values_out = tf.keras.layers.Dense(
            self.n_scenarios,
            name="values_out",
            activation="linear"
        )(obs_input)

        self.base_model = tf.keras.Model(
            [obs_input, scenario_mask],
            [action_logits, values_out])


    def forward(self, input_dict, state, seq_lens):

        obs_input = input_dict[SampleBatch.OBS]

        if SampleBatch.INFOS not in input_dict:
            self.scenario_mask = tf.one_hot([0], depth=self.n_scenarios)
        else:
            print( input_dict[SampleBatch.INFOS])
            self.scenario_mask = input_dict[SampleBatch.INFOS]

        context, self._values_out = self.base_model(
            [obs_input, self.scenario_mask]
        )
        return tf.reshape(context, [-1, self.num_outputs]), state

    def value_function(self):
        print(self.scenario_mask)
        return tf.reshape(
            tf.reduce_sum(tf.one_hot(self.scenario_mask, depth=self.n_scenarios) * self._values_out, axis=-1)
            , [-1])


    def metrics(self):

        return {
            "scenarios": tf.reduce_mean(tf.math.argmax(self.scenario_mask))
        }