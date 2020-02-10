import tensorflow as tf
import numpy as np
from typing import Union


def adv_perturbation_closed_form(model: tf.keras.Model,
                                 x: tf.Tensor,
                                 y: tf.Tensor,
                                 eps: float = 0.01) -> tf.Tensor:
    """
    Uses the closed form of projected descent for single-layer networks.

    :param model: keras model with a single layer of weights
    :param x: input tensor
    :param y: Tensor with true labels
    :param eps: amount to scale perturbation by
    :return: a tensor with adversarial samples
    """
    weights = model.get_weights()[0]
    y_batch_plus_minus_one = tf.where(tf.equal(y, 1.0), 1.0, -1.0)
    perturbation = -eps*y_batch_plus_minus_one[:, None] @ tf.transpose(
        tf.sign(weights))
    return x + tf.squeeze(perturbation)


def clip_eta(eta: tf.Tensor, norm, eps: float):
    """
    Helper function to clip the perturbation to epsilon norm ball.

    :param eta: A tensor with the current perturbation.
    :param norm: Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param eps: Epsilon, bound of the perturbation.
    """

    # Clipping perturbation eta to self.norm norm ball
    if norm not in [np.inf, 1, 2]:
        raise ValueError('norm must be np.inf, 1, or 2.')
    axis = list(range(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("")
            # This is not the correct way to project on the L1 norm ball:
            # norm = tf.maximum(avoid_zero_div, reduce_sum(tf.abs(eta),
            # reduc_ind, keepdims=True))
        elif norm == 2:
            # avoid_zero_div must go inside sqrt to avoid a divide by zero in
            # the gradient through this operation
            norm = tf.sqrt(
                tf.maximum(avoid_zero_div,
                           tf.reduce_sum(tf.square(eta), axis, keepdims=True)))
        # We must *clip* to within the norm ball, not *normalize* onto the
        # surface of the ball
        factor = tf.minimum(1., tf.math.divide(eps, norm))
        eta = eta*factor
    return eta


def adv_perturbation_pgd(model: Union[tf.Module, tf.keras.Model],
                         x: tf.Tensor,
                         eps: float,
                         eps_iter: float,
                         nb_iter: int,
                         norm,
                         clip_min: float = None,
                         clip_max: float = None,
                         y: tf.Tensor = None,
                         targeted: bool = False,
                         rand_init=None,
                         rand_minmax: float = 0.3,
                         sanity_checks: bool = True) -> tf.Tensor:
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to 0. or the
    Madry et al. (2017) method when rand_minmax is larger than 0.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf

    :param model: a callable that takes an input tensor and returns logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see
                https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf,
                1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial
                      example components.
    :param clip_max: (optional) float. Maximum float value for adversarial
                      example components.
    :param y: (optional) Tensor with true labels. If targeted is true,
              then provide the
              target label. Otherwise, only provide this parameter if you'd
              like to use true
              labels when crafting adversarial samples. Otherwise,
              model predictions are used
              as labels to avoid the "label leaking" effect (explained in
              this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being
              more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to
              use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """

    assert eps_iter <= eps, (eps_iter, eps)
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 "
            "FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature.")
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    # Initialize loop variables
    if rand_init:
        rand_minmax = eps
        eta = tf.random.uniform(x.shape, -rand_minmax, rand_minmax)
    else:
        eta = tf.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if y is None or not targeted:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model(x), 1)

    i = 0
    while i < nb_iter:
        adv_x = fast_gradient_method(model,
                                     adv_x,
                                     eps_iter,
                                     norm,
                                     clip_min=clip_min,
                                     clip_max=clip_max,
                                     y=y,
                                     targeted=targeted)

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


def fast_gradient_method(model: Union[tf.Module, tf.keras.Model],
                         x: tf.Tensor,
                         eps: float,
                         norm,
                         clip_min: float = None,
                         clip_max: float = None,
                         y: tf.Tensor = None,
                         targeted: bool = False,
                         sanity_checks: bool = False) -> tf.Tensor:
    """
    Tensorflow 2.0 implementation of the Fast Gradient Method.

    :param model: a callable that takes an input tensor and returns the
                  model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see
                https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf,
                  1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial
                    example components.
    :param clip_max: (optional) float. Maximum float value for adversarial
                      example components.
    :param y: (optional) Tensor with true labels. If targeted is true,
              then provide the
              target label. Otherwise, only provide this parameter if you'd
              like to use true
              labels when crafting adversarial samples. Otherwise,
              model predictions are used
              as labels to avoid the "label leaking" effect (explained in
              this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being
              more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to
              use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be either np.inf, 1, or 2.")

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    if y is None or not targeted:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model(x), 1)

    grad = compute_gradient(model, x, y, targeted)

    optimal_perturbation = optimize_linear(grad, eps, norm)
    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


@tf.function
def compute_gradient(model: Union[tf.Module, tf.keras.Model], x: tf.Tensor,
                     y: tf.Tensor,
                     targeted: bool) -> tf.Tensor:
    """
    Computes the gradient of the loss with respect to the input tensor.

    :param model: a callable that takes an input tensor and returns the
                  model logits.
    :param x: input tensor
    :param y: Tensor with true labels. If targeted is true, then provide the
              target label.
    :param targeted:  bool. Is the attack targeted or untargeted?
                      Untargeted, the default, will
                      try to make the label incorrect. Targeted will instead
                      try to move in the
                      direction of being more like y.
    :return: A tensor containing the gradient of the loss with respect to
            the input tensor.
    """
    loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
    with tf.GradientTape() as g:
        g.watch(x)
        # Compute loss
        loss = loss_fn(labels=y, logits=model(x))
        if targeted:  # attack is targeted, minimize loss of target label
            # rather than maximize loss of correct label
            loss = -loss

    # Define gradient of loss wrt input
    grad = g.gradient(loss, x)
    return grad


def optimize_linear(grad: tf.Tensor, eps: float, norm=np.inf) -> tf.Tensor:
    """
    Solves for the optimal input to a linear function under a norm constraint.
    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: tf tensor containing a batch of gradients
    :param eps: float scalar specifying size of constraint region
    :param norm: int specifying order of norm
    :returns: tf tensor containing optimal perturbation
    """

    # Convert the iterator returned by `range` into a list.
    axis = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = tf.sign(grad)
        # The following line should not change the numerical results. It
        # applies only because
        # `optimal_perturbation` is the output of a `sign` op, which has zero
        # derivative anyway.
        # It should not be applied for the other norms, where the perturbation
        # has a non-zero derivative.
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif norm == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, axis, keepdims=True)
        tied_for_max = tf.dtypes.cast(tf.equal(abs_grad, max_abs_grad),
                                      dtype=tf.float32)
        num_ties = tf.reduce_sum(tied_for_max, axis, keepdims=True)
        optimal_perturbation = sign*tied_for_max/num_ties
    elif norm == 2:
        square = tf.maximum(
            avoid_zero_div, tf.reduce_sum(tf.square(grad), axis,
                                          keepdims=True))
        optimal_perturbation = grad/tf.sqrt(square)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are currently implemented.")

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation