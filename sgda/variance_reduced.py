from typing import Dict, Tuple, Union, List

from ray.rllib import TFPolicy, SampleBatch
from ray.rllib.utils import try_import_tf, force_list
from ray.rllib.utils.typing import ModelGradients, TensorType, LocalOptimizer

tf1, tf, tfv = try_import_tf()


class L_SVRGDA(tf1.train.SGD):

    def __init__(self, update_prob, learning_rate=1e-3):
        self.update_prob = update_prob
        self.w = None
        super().__init__(learning_rate=learning_rate, name="L_SVRGDA")

    # def _apply_dense(self, grad, var):
    #
    #     return super()._apply_dense(grad, var)
    #
    # def _resource_apply_dense(self, grad, handle):
    #     return gen_training_ops.resource_apply_gradient_descent(
    #         handle.handle, math_ops.cast(self._learning_rate_tensor,
    #                                      grad.dtype.base_dtype),
    #         grad, use_locking=self._use_locking)
    #
    # def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    #     return resource_variable_ops.resource_scatter_add(
    #         handle.handle,
    #         indices,
    #         -grad * math_ops.cast(self._learning_rate_tensor,
    #                               grad.dtype.base_dtype))
    #
    # def _apply_sparse_duplicate_indices(self, grad, var):
    #     delta = indexed_slices.IndexedSlices(
    #         grad.values *
    #         math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
    #         grad.indices, grad.dense_shape)
    #     return var.scatter_sub(delta, use_locking=self._use_locking)
    #
    # def _prepare(self):
    #     learning_rate = self._call_if_callable(self._learning_rate)
    #     self._learning_rate_tensor = ops.convert_to_tensor(
    #         learning_rate, name="learning_rate")


class L_SVRGDA_POLICY(TFPolicy):

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(
            *args,
            **kwargs
        )

        self.optim_wrapper = L_SVRGDA(update_prob=self.config["L_SVRGDA/update_prob"])

    def optimizer(self) -> "tf.keras.optimizers.Optimizer":

        if hasattr(self, "config") and "lr" in self.config:
            return tf1.train.SGD(learning_rate=self.config["lr"])
        else:
            return tf1.train.SGD()


