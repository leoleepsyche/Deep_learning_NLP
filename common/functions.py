class functions():
    def softmax(a):
        c = np.max(a)
        exp_a = np.exp(a - c) # avoid overflow
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y
    def cross_entropy_error(y,t):
        if y.ndim == 1:
            t = t.reshape(1,t.size)
            y = y.reshape(1,y.size)
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + 1e-7)) / batch_size
    def numerical_gradient(f,x):
        h = 1e-4
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x)
            x[idx] = tmp_val - h
            fxh2 = f(x)
            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val
            it.iternext()
        return grad