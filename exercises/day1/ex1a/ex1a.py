from pathlib import Path

from matplotlib.pyplot import subplots
import numpy as np

from pymor.analyticalproblems.domaindescriptions import LineDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction
from pymor.discretizers.builtin import discretize_stationary_cg

domain = LineDomain(domain=[-1., 1.], left='dirichlet', right='dirichlet')
rhs = ExpressionFunction('(x < 0) * 1.', 1, ())
diffusion = ExpressionFunction('(x < 0) * 1. + (x > 0) * exp(m)', 1, (), parameters={'m': 1})
problem = StationaryProblem(domain=domain, 
                            diffusion=diffusion,
                            rhs=rhs)
m, data = discretize_stationary_cg(problem, diameter=.5**6)

U = m.solution_space.empty()
parameter_space = problem.parameters.space(1, 10)
parameters = []
for mu in parameter_space.sample_uniformly(10):
    parameters.append(mu['m'][0])
    U.append(m.solve(mu))

stretch, offset = data['grid'].embeddings(0)
x = np.concatenate([offset[:, 0], stretch[-1, 0] + offset[-1]])

fig, ax = subplots()
for mu, u in zip(parameters, U.to_numpy()):
    ax.plot(x, u, label=mu)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u$')
ax.legend()
fig.savefig(Path(__file__).with_suffix('.png'))
