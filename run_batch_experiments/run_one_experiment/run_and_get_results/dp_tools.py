"""
Differential Privacy Tool Module

"""

import torch
import torch.nn as nn
import numpy as np
import copy


def params_norm(params, norm_type: float = 2.0, error_if_nonfinite: bool = False) -> torch.Tensor:
    """
    Calculate the norm of parameter tensors
    """
    norm_type = float(norm_type)
    if len(params) == 0:
        return torch.tensor(0.)
    device = params[0].device
    if norm_type == np.inf:
        norms = [p.detach().abs().max().to(device) for p in params]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        all_param_norm = torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in params])
        total_norm = torch.norm(all_param_norm, norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    return total_norm, all_param_norm


def proj_by_norm_(parameters, min_norm, max_norm, norm_type=2):
    """
    Project (clip) parameters based on norm
    """
    # 1. calc grad norm
    total_norm, _ = params_norm(parameters, norm_type=norm_type)

    # 2. calc norm-based scaling factor
    min_norm = torch.Tensor([min_norm]).to(total_norm.device)
    max_norm = torch.Tensor([max_norm]).to(total_norm.device)
    if total_norm < min_norm:
        coef = min_norm / (total_norm + 1e-9)
    else:
        coef = torch.clamp(max_norm / total_norm, max=1.0)

    # 3. apply grad scaling
    for p in parameters:
        p.detach().mul_(coef.to(p.device))

    return total_norm, coef


def gaussian_noise(data_shape, sigma, device=None):
    """
    Generate Gaussian noise
    """
    return torch.normal(0, sigma, data_shape).to(device)


def laplace_noise(data_shape, scale, device=None):
    """
    Generate Laplace noise
    """
    m = torch.distributions.laplace.Laplace(torch.tensor(0.0), torch.tensor(scale))
    return m.sample(data_shape).to(device)


def dp_scale_laplace(eps, clip, lr):
    """
    Calculate the scale for Laplace noise
    """
    # Ensure eps is a float
    eps = float(eps)
    sens = 2 * clip * lr
    scale = sens / eps
    return scale


def add_dp_noise_to_gradients(model, clip, eps, mechanism='laplace', clip_level='batch', element_wise_rand=True, lr=0.01):

    assert mechanism in ['laplace', 'gaussian']
    assert clip_level in ['sample', 'batch']
    
    # Ensure eps is a float
    eps = float(eps)
    
    clip_norm_type = 1 if mechanism == 'laplace' else 2
    
    # Collect all gradients
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad)
    
    if not gradients:
        return
    
    # Gradient clipping
    if clip_level == 'sample':
        # Simplified handling here; actual implementation should clip per sample
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip, norm_type=clip_norm_type)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip, norm_type=clip_norm_type)
    
    # Add noise
    sens = 2 * clip * lr
    scale = sens / eps
    
    for param in model.parameters():
        if param.grad is not None:
            if element_wise_rand:
                if mechanism == 'laplace':
                    noise = laplace_noise(param.grad.shape, scale, device=param.grad.device)
                else:  # gaussian
                    noise = gaussian_noise(param.grad.shape, scale, device=param.grad.device)
            else:
                if mechanism == 'laplace':
                    noise = laplace_noise([1], scale, device=param.grad.device) * torch.ones_like(param.grad)
                else:  # gaussian
                    noise = gaussian_noise([1], scale, device=param.grad.device) * torch.ones_like(param.grad)
            
            param.grad.add_(noise)


class DPClipper:
    """
    Differential Privacy Clipper
    Encapsulates DP mechanisms for easy use in FedAvg
    """
    
    def __init__(self, config, logger, device):
        self.config = config
        self.logger = logger
        self.device = device
        
        # DP parameters - ensure correct numeric types
        self.dp_mechanism = config.get("dp_mechanism", "no_dp")
        self.dp_epsilon = float(config.get("dp_epsilon", 5.0))
        self.dp_delta = float(config.get("dp_delta", 1e-5))
        self.dp_clip = float(config.get("dp_clip", 1.0))
        self.dp_noise_scale = float(config.get("dp_noise_scale", 0.1))
        self.dp_clip_level = config.get("dp_clip_level", "batch")
        self.dp_element_wise_rand = config.get("dp_element_wise_rand", True)
        
        self.logger.info(f"DPClipper initialized: mechanism={self.dp_mechanism}, epsilon={self.dp_epsilon}, "
                        f"clip={self.dp_clip}, clip_level={self.dp_clip_level}")
    
    def apply_dp_to_gradients(self, model, lr=0.01):
        """
        Apply DP to model gradients
        """
        if self.dp_mechanism == "no_dp":
            return
        
        if self.dp_mechanism == "laplace":
            mechanism = 'laplace'
        elif self.dp_mechanism == "gaussian":
            mechanism = 'gaussian'
        else:
            self.logger.warning(f"Unknown DP mechanism: {self.dp_mechanism}, using laplace as default")
            mechanism = 'laplace'
        
        add_dp_noise_to_gradients(
            model=model,
            clip=self.dp_clip,
            eps=self.dp_epsilon,
            mechanism=mechanism,
            clip_level=self.dp_clip_level,
            element_wise_rand=self.dp_element_wise_rand,
            lr=lr
        )
    
    def get_privacy_budget(self):
        """
        Get privacy budget information
        """
        if self.dp_mechanism == "no_dp":
            return "No DP protection"
        
        return f"ε={self.dp_epsilon}, δ={self.dp_delta}, clip={self.dp_clip}"