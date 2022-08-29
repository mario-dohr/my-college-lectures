"""
Utility functions for checking neural network implementations.
"""
import numpy as np


def numerical_gradient(func, *inputs, eps=1e-6):
    """
    Compute numerical gradient using two-sided numerical gradient approximation.

    Parameters
    ----------
    func : callable
        Function to compute the gradient for.
        This function must not have side-effects.
    x0, x1, ..., xn : ndarray
        One or more inputs to compute the gradients for.
    eps : float, optional
        The finite difference to use in the computation.

    Returns
    -------
    grad0, grad1, ... gradn : ndarray
        The numerical gradient for every input.

    Examples
    --------
    >>> x = np.random.randn(10)
    >>> dx = numerical_gradient(lambda a: 2 * a, x)
    >>> y = np.random.randn(10)
    >>> dx, dy = numerical_gradient(lambda a, b: a * b, x, y)
    """
    if len(inputs) == 0:
        return ()

    gradients = []
    for x in inputs:
        x_flat = x.flat
        original_x_flat = x_flat.copy()

        x_grads = []
        for i in range(x.size):
            # f(x + eps)
            x_flat[i] = original_x_flat[i] + eps
            output = func(*inputs)
            f_high = np.sum(output)

            # f(x - eps)
            x_flat[i] = original_x_flat[i] - eps
            output = func(*inputs)
            f_low = np.sum(output)

            # restore weight + compute result
            x_flat[i] = original_x_flat[i]
            num_grad = (f_high - f_low) / (2 * eps)
            x_grads.append(num_grad)

        gradients.append(np.reshape(x_grads, x.shape))

    return tuple(gradients) if len(inputs) > 1 else gradients[0]


def gradient_check(module, *inputs, eps=1e-6, debug=False):
    """
    Compare analytical gradients with numerical gradient approximation.

    Parameters
    ----------
    module : Module
        Numpy deep learning module with parameters.
    x0, x1, ..., xn : ndarray
        Input data to check gradients on.
    eps : float, optional
        The finite difference to use for numerical gradient computation.
    debug : bool, optional
        Flag to print gradients for debugging.

    Examples
    --------
    >>> fc = nnumpy.Linear(5, 2)
    >>> x = np.random.randn(3, 5)
    >>> gradient_check(fc, x, debug=True)
    """
    # discard parameters in state to save copy time
    original_state = module.get_state(parameters=False)

    # analytical gradients
    module.zero_grad()
    pred = module.forward(*inputs)
    dx_analytic = module.backward(np.ones_like(pred))
    dw_analytic = tuple(w.grad for w in module.parameters())

    # numerical input gradients
    def stateless_inputs_func(*xs):
        """ module as function of inputs without side-effects. """
        module.set_state(original_state)
        y, _ = module.compute_outputs(*xs)
        return y

    dx_numeric = numerical_gradient(stateless_inputs_func, *inputs, eps=eps)
    dx_check = np.allclose(_cat(dx_numeric), _cat(dx_analytic), atol=2 * eps)

    if debug and not dx_check:
        print(np.abs(_cat(dx_numeric) - _cat(dx_analytic)).max(), '>', 2 * eps)
        print("dx numeric: ", dx_numeric)
        print("dx analytic:", dx_analytic)

    # numerical parameter gradients
    def stateless_parameter_func(*pars):
        """ module as function of parameters without side-effects. """
        # parameters will be changed in-place
        module.set_state(original_state)
        y, _ = module.compute_outputs(*inputs)
        return y

    params = tuple(module.parameters())
    dw_numeric = numerical_gradient(stateless_parameter_func, *params, eps=eps)
    dw_check = np.allclose(_cat(dw_numeric), _cat(dw_analytic), atol=2 * eps)

    if debug and not dw_check:
        for num_grad, (name, par) in zip(dw_numeric, module.named_parameters()):
            err = np.abs(num_grad - par.grad).max()

            if err > 2 * eps:
                print(err, '>', 2 * eps)
                print(f"d{name} numeric: ", num_grad)
                print(f"d{name} analytic:", par.grad)

    return dx_check and dw_check


def _cat(arrs):
    """ Concatenate numpy arrays with different numbers of dimensions. """
    if len(arrs) == 0:
        return np.array([])

    try:
        # NOTE: will fail when `arrs` is not an ndarray (safety check)
        return arrs.ravel()
    except AttributeError:
        return np.concatenate([arr.ravel() for arr in arrs])
