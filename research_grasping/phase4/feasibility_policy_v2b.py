"""Archive-prevalence, dimension-normalized feasibility smoother (experimental)."""
import numpy as np
from axisforge_bo.models import ExactGP, StationaryKernel
from research_grasping.phase4.feasibility_policy import FeasibilityPolicy, kernel_probability


class PrevalenceFeasibilityPolicy(FeasibilityPolicy):
    """Target or adaptive control sharing train-derived object-balanced prior."""
    def __init__(self, *args, prior_strength=2., quality_length_scale=1., **kwargs):
        history=kwargs.get('history') if 'history' in kwargs else args[3]
        if not np.isfinite(prior_strength) or prior_strength <= 0:
            raise ValueError('invalid prior strength')
        super().__init__(*args, **kwargs)
        rates=[float(np.mean(task.feasible)) for task in history]
        self.kernel=StationaryKernel(length_scale=quality_length_scale)
        self.quality=ExactGP(self.kernel, self.noise)
        self.prior_mean=(1.+sum(rates))/(2.+len(rates)) if rates else .5
        self.alpha=prior_strength*self.prior_mean
        self.beta=prior_strength*(1-self.prior_mean)

    def probabilities(self,x):
        length=self.length_scale*np.sqrt(self.x.shape[1])
        parts=[kernel_probability(x,self.obs_x,self.labels,length_scale=length,alpha=self.alpha,beta=self.beta)]
        parts.extend(kernel_probability(x,tx,y,length_scale=length,alpha=self.alpha,beta=self.beta)
                     for _,_,tx,y in self.sources)
        return np.array(parts)
