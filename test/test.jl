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

χ = 900;
mₚ = [1.0, 10.0, 3.0];
A = diagm([500.0, 0.0002, 200.0]);
b = h = [100.0, 1.0, 0.0];
a = 10
m = 1e9

S(u) = u' * A * u + b' * u
#S(u) = 2 / π * m * norm(u) * atan(norm(u) / a) + 1 / 2 * log(a^2 + norm(u)^2)
#S(u) = log(exp(u[1])+ exp(u[2]) + 1)

prob = ConstrainedProblem(χ, h, mₚ, S)
intf = Interface(prob, h, 10000, 10e-10)
sol1 = solve(intf, :newton, linesearch = BackTracking(order = 2))
#sol2 = solve(intf, :newton, linesearch=HagerZhang())
sol3 = solve(intf, :newton, linesearch = StrongWolfe())
#sol4 = solve(intf, :newton, linesearch=ls)
sol5 = solve(intf, :newton, linesearch = Static())

sol5.sol - h |> norm

@profview [solve(intf, :newton, linesearch = StrongWolfe()) for i in 1:100]

surface(LinRange(-π, π, 100), LinRange(-π, π, 100), (x, y) -> intf.prob.obj(transform_to_euklidean_3D([x, y], χ, h)))

