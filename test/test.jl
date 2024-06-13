using Revise
using BenchmarkTools
using Bachelorarbeit
using LinearAlgebra
using Profile, ProfileView
using LineSearches

using Zygote

function ls(ϕ, dϕ, ϕdϕ, α, fk, dϕ₀)
    i = 0
    while ϕ(α) - fk ≤ 0.5 * α * dϕ₀
        i += 1
        if (i > 10.000)
            throw(error("Maximum Iteration reached"))
        end
        α = α * 0.99
    end
    α, ϕ(α)
end

χ = 900;
h = mₚ = [1.0, 4.0];
A = diagm([5.0, 0.02]);
b = [-0.3, 0.3];

#S(u) = u' * A * u + b' * u
#S(u) = norm(u) * atan(norm(u)) + log(1 + norm(u)^2)
#S(u) = log(exp(u[1])+ exp(u[2]) + 1)

prob = ConstrainedProblem(χ, h, mₚ, S)
intf = Interface(prob, h, 100000, 10e-6)
sol1 = solve(intf, :newton, linesearch=BackTracking())
sol2 = solve(intf, :newton, linesearch=HagerZhang())
sol3 = solve(intf, :newton, linesearch=StrongWolfe())
sol4 = solve(intf, :newton, linesearch=ls)
sol5 = solve(intf, :newton, linesearch=Static())

sol5.sol - h |> norm

sol3.f
using Plots
plot(range(-π, π, 100), x -> intf.prob.obj(transform_to_euklidean_2D([x], χ, h)))

surface(range(-1, 10, 100), range(-1, 10, 100), (x, y) -> intf.prob.obj([x, y]))