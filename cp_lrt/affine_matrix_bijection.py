import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


class AffineLinearOperatorMatrix(tfp.python.bijectors.Bijector):
    """\
    Extend AffineLinearOperator to work with matrix inputs

    """
    def __init__(self, shift=None, scale=None, adjoint=False, validate_args=False, name='affine_linear_operator_matrix'):
        self._graph_parents = []
        self._name = name
        self._validate_args = validate_args
        graph_parents = []
        with self._name_scope("init", values=[shift]):
            # In the absence of `loc` and `scale`, we'll assume `dtype` is `float32`.
          dtype = tf.float32

          if shift is not None:
            shift = tf.convert_to_tensor(shift, name="shift")
            graph_parents += [shift]
            dtype = shift.dtype.base_dtype
          self._shift = shift

          if scale is not None:
            if (shift is not None and
                shift.dtype.base_dtype != scale.dtype.base_dtype):
              raise TypeError(
                  "shift.dtype({}) is incompatible with scale.dtype({}).".format(
                      shift.dtype, scale.dtype))
            if not isinstance(scale, tf.linalg.LinearOperator):
              raise TypeError("scale is not an instance of tf.LinearOperator")
            if validate_args and not scale.is_non_singular:
              raise ValueError("Scale matrix must be non-singular.")
            graph_parents += scale.graph_parents
            if scale.dtype is not None:
              dtype = scale.dtype.base_dtype
          self._scale = scale
          self._adjoint = adjoint
          super(AffineLinearOperatorMatrix, self).__init__(
              forward_min_event_ndims=1,
              graph_parents=graph_parents,
              is_constant_jacobian=True,
              dtype=dtype,
              validate_args=validate_args,
              name=name)

    @property
    def shift(self):
        """The `shift` `Tensor` in `Y = scale @ X + shift`."""
        return self._shift

    @property
    def scale(self):
        """The `scale` `LinearOperator` in `Y = scale @ X + shift`."""
        return self._scale

    @property
    def adjoint(self):
        """`bool` indicating `scale` should be used as conjugate transpose."""
        return self._adjoint

    def _forward(self, X):
        Y = X
        if self.scale is not None:
            with tf.control_dependencies(self._maybe_collect_assertions()
                                   if self.validate_args else []):
                Y = self.scale.matmul(Y, adjoint=self.adjoint)  # change here
        if self.shift is not None:
            Y += self.shift
        return Y

    def _inverse(self, Y):
        X = Y
        if self.shift is not None:
            X -= self.shift
        if self.scale is not None:
            # Solve fails if the op is singular so we may safely skip this assertion.
            X = self.scale.solve(X, adjoint=self.adjoint)
        return X

    def _forward_log_det_jacobian(self, X):
        # is_constant_jacobian = True for this bijector, hence the
        # `log_det_jacobian` need only be specified for a single input, as this will
        # be tiled to match `event_ndims`.
        if self.scale is None:
            return tf.constant(0., dtype=x.dtype.base_dtype)

        with tf.control_dependencies(self._maybe_collect_assertions()
                                    if self.validate_args else []):
            return self.scale.log_abs_determinant()

    def _maybe_collect_assertions(self):
        try:
            return [self.scale.assert_non_singular()]
        except NotImplementedError:
            pass
        return []