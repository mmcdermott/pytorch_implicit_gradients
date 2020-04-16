import torch

class IterativeConjugateGradientEngine:
    """
    https://arxiv.org/pdf/1909.04630.pdf
    Inspired by https://github.com/sungyubkim/GBML/blob/master/gbml/imaml.py

    Only works to compute \partial LV / \parital \phi^{(0)}, where LV is validation
    loss under optimal parameters \phi^* computed via SGD with proximal regularization
    on loss LT with weighting $\lambda$.
    """
    def __init__(self, proximal_regularization_strength):
        self.pr_lambda = proximal_regularization_strength

    @torch.no_grad()
    def cg(self, in_grad, outer_grad, params):
        x = outer_grad.clone().detach()
        r = outer_grad.clone().detach() - self.hv_prod(in_grad, x, params)
        p = r.clone().detach()
        for i in range(self.n_cg):
            Ap = self.hv_prod(in_grad, p, params)
            alpha = (r @ r)/(p @ Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = (r_new @ r_new)/(r @ r)
            p = r_new + beta * p
            r = r_new.clone().detach()
        return self.vec_to_grad(x, params)

    # TODO(mmd): Port this logic over to all algorithms.
    def vec_to_grad(self, vec, params):
        pointer = 0
        res = []
        for param in params:
            num_param = param.numel()
            res.append(vec[pointer:pointer+num_param].view_as(param).data)
            pointer += num_param
        return res

    @torch.enable_grad()
    def hv_prod(self, in_grad, x, params):
        hv = torch.autograd.grad(in_grad, params, retain_graph=True, grad_outputs=x)
        hv = torch.nn.utils.parameters_to_vector(hv).detach()
        # precondition with identity matrix
        return hv/self.pr_lambda + x


    def implicit_grad(
        val_grad, train_loss, train_params, meta_params = None, learning_rate = None,
        direct_val_meta_grad = None,
    ):
        if meta_params is not None:
            assert len(list(train_params)) == len(list(meta_params))
            print(
                "Meta parameters not actually needed for this algorithm, "
                "as they should be the initialization of the `train_params`."
            )

        train_grad = torch.nn.utils.parameters_to_vector(
            torch.autograd.grad(train_loss, train_params, create_graph=True)
        )
        implicit_grad = self.cg(train_grad, val_grad, train_params)
        return implicit_grad

class NeumannInverseHessianApproximationEngine:
    """
    https://arxiv.org/pdf/1911.02590v1.pdf


    """
    def __init__(self, num_inverse_hvp_iters, allow_unused=False):
        self.num_inverse_hvp_iters = num_inverse_hvp_iters
        self.allow_unused = allow_unused

    def approx_inverse_HVP(self, v, f, w, alpha):
        """
        Algorithm 3 from https://arxiv.org/pdf/1911.02590v1.pdf
        w is expected to be pytorch parameters.
        f the output of a computation graph dependening on w.
        alpha should be the learning rate of the update process for w.

        Computes approximation of inverse-Hessian-vector product
        v[\partial f/ \partial w]^{-1}
        """
        p = v
        for j in range(self.num_inverse_hvp_iters):

            diff = torch.autograd.grad(
                f, w, grad_outputs=v, retain_graph=True, allow_unused=self.allow_unused
            )
            assert len(list(diff)) == len(list(v)), (
                "Length mismatch! len(diff) = %d, len(v) = %d, len(w) = %d"
                "" % (len(list(diff)), len(list(v)), len(list(w)))
            )
            v = [ve - alpha * ge for ve, ge in zip(v, diff)]
            p = [pe + ve for pe, ve in zip(p, v)]
        return [alpha*pe for pe in p]

    def implicit_grad(
        self, val_grad, train_loss, train_params, meta_params, learning_rate,
        direct_val_meta_grad = None,
    ):
        """
        Runs the full implicit (indirect) gradient computation from https://arxiv.org/pdf/1911.02590v1.pdf.

        val_grad: v1 in algorithm 2. Should be \partial L_V / \partial \phi |_{\phi^*}.
        train_loss: L_T, the loss optimized during training to produce \phi^* via SGD.
        train_params: \phi, the training parameters.
        meta_params: \theta, the meta-parameters.
        learning_rate: \alpha, the learning rate for the SGD process to produce \phi^*
        direct_val_meta_grad: \partial L_V / \partial \theta (this is typically zero)

        TODO: May be able to simplify this somewhat, specifically for common relationships like proximal regularization.
        """
        if direct_val_meta_grad is not None:
            # TODO: check grad properly
            assert len(list(indirect_val_meta_grad)) == len(list(direct_val_meta_grad))


        train_grad = torch.autograd.grad(
            train_loss, train_params, create_graph = True, allow_unused=self.allow_unused
        )
        inv_HVP = self.approx_inverse_HVP(val_grad, train_grad, train_params, learning_rate)
        indirect_val_meta_grad = [-ge for ge in torch.autograd.grad(
            train_grad, meta_params, grad_outputs = inv_HVP, allow_unused=self.allow_unused
        )]

        if direct_val_meta_grad is None:
            return indirect_val_meta_grad
        else:
            return [dg + ig for dg, ig in zip(direct_val_meta_grad, indirect_val_meta_grad)]
