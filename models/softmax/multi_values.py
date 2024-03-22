from typing import List

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.models import ModelV2
from ray.rllib.models.tf import TFModelV2
from ray.rllib.utils import override

class MultiValueSoftmax(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **custom_args):
        model_config.update(**custom_args)

        self.num_outputs = action_space.n
        self.n_scenarios = model_config["n_scenarios"]

        super(TFModelV2, self).__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )

        obs_input = tf.keras.layers.Input(shape=obs_space.shape, name="obs_input", dtype=tf.float32)
        scenario_mask = tf.keras.layers.Input(shape=(self.n_scenarios,), name="scenario_mask", dtype=tf.float32)

        action_out = tf.keras.layers.Dense(
            self.num_outputs,
            name="action_logits",
            activation="linear"
        )(obs_input)

        value_out = tf.reduce_sum(tf.keras.layers.Dense(
            self.n_scenarios,
            name="action_logits",
            activation="linear"
        )(obs_input) * scenario_mask, axis=-1)

        self.base_model = tf.keras.Model(
            [obs_input],
            [action_out, value_out])

    def forward(self, input_dict, state, seq_lens):

        obs_input = input_dict[SampleBatch.OBS]
        scenario_mask = input_dict[SampleBatch.INFOS]

        context, self._value_out = self.base_model(
            [obs_input, scenario_mask]
        )
        return tf.reshape(context, [-1, self.num_outputs]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])