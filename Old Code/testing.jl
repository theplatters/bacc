include("wrapper.jl")
using .Hysteresis
using ForwardDiff
using LinearAlgebra
χ = 0.01;
h = mₚ = [1.0, 2.0, 3.0];
A = diagm([5, 3, 0.001]);
b = [-0.3, 0.3, 2.0];
S(u) = (u' * A * u) + b' * u + 5;
∂S(u) = A * u + b
∂²S(u) = A

U(m) = 0.5 * (m - b)' * inv(A) * (m - b) - 5;

x0 = [1.0, 2.0, 3.01];

prob = RestrainedProblem(χ, h, mₚ, U);

uprob = UnrestrainedProblem(χ, h, mₚ, S);

intf1 = Interface(uprob, x0);
inftf2 = Interface(prob, x0);

sol1 = solve(intf1, maxiter=100, tol=10^-10,algorithm = :semiSmoothNewton)

sol2 = solve(inftf2, maxiter=100);


sol1.xk
sol2.xk

sol1

hesNorm(m) = (1/norm(m - mₚ) * I - 1/norm(m - mₚ)^3 * (m - mₚ) * (m - mₚ)')
transformToRadial(x, m) = [atan(x[2] - m[2], x[1] - m[1]), acos((x[3] - m[3]) / norm(x - m))]