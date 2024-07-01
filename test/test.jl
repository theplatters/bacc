using Revise
using BenchmarkTools, Plots, Profile, Zygote, LinearAlgebra, LineSearches
using Bachelorarbeit
plotlyjs()
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

χ = 5e1;
mₚ = [1.0, -30.0, 3.03];
A = diagm([5.0, 2.0, 1]);
b = h = [10.0, 52.0, 3.0];
a = 10
m = 1e10

S(u) = u' * A * u + b' * u
#S(u) = 2 / π * m * norm(u) * atan(norm(u) / a) + 1 / 2 * log(a^2 + norm(u)^2)
#S(u) = log(exp(u[1])+ exp(u[2]) + 1)

prob = ConstrainedProblem(χ, h, mₚ, S)
intf = Interface(prob, h, 1000, 10e-8)

sol1 = solve(intf, :semi_smooth_newton, linesearch = StrongWolfe())
sol1 = solve(intf, :newton, linesearch = StrongWolfe())
#sol2 = solve(intf, :newton, linesearch=HagerZhang())
sol3 = solve(intf, :proximal_gradient, linesearch = StrongWolfe())
#sol4 = solve(intf, :newton, linesearch=ls)
@benchmark (
	intf = Interface(prob, ones(3) + rand(3), 1000, 10e-8),
	sol5 = solve(intf, :semi_smooth_newton, linesearch = StrongWolfe()),
)

sol1.sol - sol3.sol

sol1.sol - h |> norm

@profview [solve(intf, :subgradientdescent, linesearch = StrongWolfe()) for i in 1:100]

surface(LinRange(-π, π, 100), LinRange(-π, π, 100), (x, y) -> intf.prob.obj(transform_to_euklidean_3D([x, y], χ, h)))

