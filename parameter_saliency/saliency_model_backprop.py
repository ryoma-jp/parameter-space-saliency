import time
from typing import Any, Dict, Optional

import torch
import torch.autograd as autograd
import torch.nn as nn

from utils import AverageMeter
from target.spec import TargetSpec

class SaliencyModel(nn.Module):
    """Computes parameter-space saliency scores via backpropagation.

    The objective (loss) is delegated to a TaskAdapter, which makes this
    class task-agnostic.

    Args:
        net:          The model (may be DataParallel-wrapped).
        task_adapter: Provides build_objective(model, inputs, labels, spec).
        device:       'cuda' or 'cpu'.
        mode:         'naive' | 'std' | 'norm'
        aggregation:  'filter_wise' | 'parameter_wise'
        signed:       Keep gradient sign when True.
    """

    def __init__(
        self,
        net,
        task_adapter,
        device: str = 'cuda',
        mode: str = 'std',
        aggregation: str = 'filter_wise',
        signed: bool = False,
    ):
        super(SaliencyModel, self).__init__()
        self.net          = net
        self.task_adapter = task_adapter
        self.device       = device
        self.mode         = mode
        self.aggregation  = aggregation
        self.signed       = signed

    def forward(
        self,
        inputs: torch.Tensor,
        true_labels: torch.Tensor,
        target_spec: TargetSpec,
        testset_mean_abs_grad: torch.Tensor = None,
        testset_std_abs_grad:  torch.Tensor = None,
        objective_context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        inputs.requires_grad_()
        self.net.eval()
        inputs, true_labels = inputs.to(self.device), true_labels.to(self.device)
        self.net.zero_grad()

        loss, _ = self.task_adapter.build_objective(
            self.net,
            inputs,
            true_labels,
            target_spec,
            objective_context=objective_context,
        )

        gradients = autograd.grad(
            loss, self.net.parameters(),
            create_graph=True, allow_unused=True,
        )

        filter_grads = []
        for grad in gradients:
            if grad is None:        # unused parameter — skip safely
                continue
            if self.aggregation == 'filter_wise':
                if len(grad.size()) == 4:   # Conv2d weight
                    if self.signed:
                        agg = grad.mean(-1).mean(-1).mean(-1)
                    else:
                        agg = grad.abs().mean(-1).mean(-1).mean(-1)
                    filter_grads.append(agg)
            elif self.aggregation == 'parameter_wise':
                flat = grad.view(-1)
                filter_grads.append(flat if self.signed else flat.abs())
            elif self.aggregation == 'tensor_wise':
                raise NotImplementedError("tensor_wise aggregation is not yet implemented.")

        naive_saliency = torch.cat(filter_grads)

        if self.mode == 'naive':
            return naive_saliency

        if self.mode == 'std':
            # Clone to avoid mutating the cached statistics tensor in-place
            std = testset_std_abs_grad.clone().to(self.device)
            std[std <= 1e-14] = 1.0
            return (naive_saliency - testset_mean_abs_grad.to(self.device)) / std

        if self.mode == 'norm':
            mean = testset_mean_abs_grad.clone().to(self.device)
            mean[mean <= 1e-14] = 1.0
            return naive_saliency / mean

        raise ValueError(f"Unknown saliency mode: '{self.mode}'. "
                         "Choose 'naive', 'std', or 'norm'.")


def find_testset_saliency(net, testset, aggregation, task_adapter, target_spec, signed=False):
    """Compute mean and std of filter-wise saliency across the testset.

    Uses Welford's online algorithm to update mean and variance
    incrementally, avoiding excessive memory allocation.

    Args:
        net:          The model.
        testset:      A dataset compatible with DataLoader.
        aggregation:  Aggregation strategy (see SaliencyModel).
        task_adapter: Task adapter that provides the objective function.
        target_spec:  Target specification used for gradient computation.
        signed:       Whether to keep gradient signs.

    Returns:
        (testset_mean_abs_grad, testset_std_abs_grad) as float64 tensors.
    """
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=2)

    saliency_model = SaliencyModel(
        net, task_adapter,
        device=device, mode='naive',
        aggregation=aggregation, signed=signed,
    )

    iter_time = AverageMeter()
    end       = time.time()
    testset_mean_abs_grad = None
    testset_std_abs_grad  = None

    for batch_idx, (testset_inputs, testset_targets) in enumerate(testloader):
        testset_inputs  = testset_inputs.to(device)
        testset_targets = testset_targets.to(device)

        testset_grad = saliency_model(
            testset_inputs, testset_targets, target_spec
        ).detach().double().to(device)

        if batch_idx == 0:
            testset_mean_abs_grad_prev = torch.zeros_like(testset_grad)
            testset_mean_abs_grad      = testset_grad.clone()
            testset_std_abs_grad       = (
                (testset_grad - testset_mean_abs_grad)
                * (testset_grad - testset_mean_abs_grad_prev)
            )
        else:
            testset_mean_abs_grad_prev  = testset_mean_abs_grad.detach().clone()
            testset_mean_abs_grad      += (
                (testset_grad - testset_mean_abs_grad) / float(batch_idx + 1)
            )
            testset_std_abs_grad       += (
                (testset_grad - testset_mean_abs_grad)
                * (testset_grad - testset_mean_abs_grad_prev)
            )

        iter_time.update(time.time() - end)
        end = time.time()
        if (batch_idx + 1) % 50 == 0:
            remain  = (len(testloader) - batch_idx - 1) * iter_time.avg
            h, rem  = divmod(remain, 3600)
            m, s    = divmod(rem, 60)
            print(
                f"Iter: [{batch_idx + 1}/{len(testloader)}]  "
                f"iter_time: {iter_time.val:.3f}  "
                f"remain: {int(h):02d}:{int(m):02d}:{int(s):02d}"
            )

    # Finalise variance → std  (unbiased estimator)
    testset_std_abs_grad = testset_std_abs_grad / float(len(testloader) - 1)
    testset_std_abs_grad = torch.sqrt(testset_std_abs_grad)

    print('Std:   ', testset_std_abs_grad)
    print('Mean:  ', testset_mean_abs_grad)
    print('Shape: ', testset_mean_abs_grad.shape)

    return testset_mean_abs_grad, testset_std_abs_grad


if __name__ == '__main__':
    pass