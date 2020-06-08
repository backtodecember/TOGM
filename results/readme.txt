1: FD SNOPT ,Stand in place, x move forward 0.01 proper initialization, total time = 0.01s, N = 3. Took time = 228 sec.
opt_tol = 1e-4
SNOPTA EXIT   0 -- finished successfully
SNOPTA INFO   1 -- optimality conditions satisfied

same condtion, with SA,  7 secs
SNOPTA EXIT   0 -- finished successfully
SNOPTA INFO   1 -- optimality conditions satisfied
(this on not that meaningful...)

2: SA SNOPT ,Stand in place, x move forward 0.01, standing straight initialization, total time = 0.2s, N = 41. Took time = 45 minutes SA
opt_tol = 1e-4 feasible tol 5e-4
SNOPTA EXIT  30 -- resource limit error
SNOPTA INFO  31 -- iteration limit reached


3: SA SNOPT ,Stand in place, 0.01 proper initialization, total time = 0.01s, N = 3. Took time = 10 sec SA
opt_tol = 1e-4 feasible tol 1e-4
SNOPTA EXIT   0 -- finished successfully
SNOPTA INFO   1 -- optimality conditions satisfied


4: SA SNOPT ,Stand in place, 0.01 proper initialization, total time = 0.2s, N = 41. Took time =
opt_tol = 1e-4 feasible tol 1e-4

 SNOPTA EXIT  10 -- the problem appears to be infeasible
 SNOPTA INFO  13 -- nonlinear infeasibilities minimized

5: FD SNOPT ,Stand in place, 0.01 proper initialization, total time = 0.2s, N = 41. Took time = 3 hours
opt_tol = 1e-4 feasible tol 1e-4

 SNOPTA EXIT  30 -- resource limit error
 SNOPTA INFO  31 -- iteration limit reached

6: FD SNOPT ,Stand in place, 0.01 proper initialization, total time = 0.01s, N = 41. Took time = 10 mins 
opt_tol = 1e-4 feasible tol 1e-4
SNOPTA EXIT   0 -- finished successfully
SNOPTA INFO   1 -- optimality conditions satisfied
