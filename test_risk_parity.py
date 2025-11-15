print('Starting test...')
from allooptim.optimizer.sequential_quadratic_programming.risk_parity_optimizer import RiskParityOptimizer
import numpy as np
from scipy.optimize import approx_fprime

print('Imports done...')
optimizer = RiskParityOptimizer()
n = 4
x0 = np.array([0.25, 0.25, 0.25, 0.25])
cov = np.array([
    [0.04, 0.01, 0.00, 0.00],
    [0.01, 0.09, 0.00, 0.00],
    [0.00, 0.00, 0.16, 0.02],
    [0.00, 0.00, 0.02, 0.25]
])

optimizer._cov = cov
optimizer._target_risk = np.ones(n) / n

print('Setup done...')
print('Testing Risk Parity Jacobian...')
try:
    obj_val = optimizer._risk_budget_objective(x0)
    print('Objective value:', obj_val)
    ana_grad = optimizer._risk_budget_jacobian(x0)
    print('Analytical gradient shape:', ana_grad.shape)
    print('Analytical gradient:', ana_grad)
    
    print('Computing numerical gradient...')
    num_grad = approx_fprime(x0, optimizer._risk_budget_objective, epsilon=1e-6)  # Larger epsilon
    print('Numerical gradient shape:', num_grad.shape)
    print('Numerical gradient:', num_grad)
    print('Max difference:', np.max(np.abs(num_grad - ana_grad)))
    
    if np.allclose(num_grad, ana_grad, rtol=1e-2, atol=1e-4):
        print('Test passed!')
    else:
        print('Test failed - gradients do not match')
        
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()