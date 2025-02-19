using Bachelorarbeit
using LineSearches
using LinearAlgebra
using GLMakie
using Random
using Statistics

include("testfunctions.jl")

struct Params2
	χ::Float64
	h::Vector{Float64}
	mₚ::Vector{Float64}
end

function plot_convergencerate(data)
	fig = Figure(size = (1980, 1020), fontsize=24)
	ax1 = Axis(fig[1, 1], xticks = 1:maximum(size(data)), xlabel = "Iterations", ylabel = "Error", yscale = log10, title = "Unconstrained methods")
	ax2 = Axis(fig[1, 2], xticks = 1:maximum(size(data)), xlabel = "Iterations", ylabel = "Error", yscale = log10, title = "Constrained methods")
	for (l, d) in zip(["Semi-smooth Newton (unconstrained)", "Subgradient descent", "Proximal gradient"], eachrow(data[1:3, :]))
		lines!(ax1, d[d.!=-Inf], label = l)
	end
	for (l, d) in zip(["Newton", "Semi-smooth Newton (constrained)"], eachrow(data[4:5, :]))
		lines!(ax2, 1:length(d[d.!=-Inf]), d[d.!=-Inf], label = l)
	end
	axislegend(ax1, "Methods", position = :rt)
	axislegend(ax2, "Methods", position = :rt)
	fig
end

function test_problem(fun, fun_conv, p::Params2)

	max_iter = 15
	data = fill(-Inf, 5, max_iter + 1)

	up = UnconstrainedProblem(p.χ, p.h, p.mₚ, fun)
	ui = Interface(up, p.mₚ, max_iter, 1e-13)
	cp = ConstrainedProblem(p.χ, p.h, p.mₚ + [0.001, 1e-10, 1e-10], fun_conv)
	ci = Interface(cp, p.h + [0.001, 1e-10, 1e-10], max_iter, 1e-13)

	sol_uc = solve(ui, :semi_smooth_newton, linesearch = WolfePowell())
	sol_c = solve(ci, :semi_smooth_newton, linesearch = Static())

	@info sol_uc
	function calculate_error_uc!(i)
		function a(cache, _)
			data[i, 1+cache.iter] = abs(cache.fk - sol_uc.f) / abs(sol_uc.f)
		end
	end

	function calculate_error_c!(i)
		function a(cache, _)
			@info abs(cache.fk - sol_c.f)

			data[i, 1+cache.iter] = abs(cache.fk - sol_c.f) / abs(sol_c.f)
		end
	end



	data[1:3, 1] .= norm(up.oracle(p.mₚ))
	data[4:5, 1] .= norm(cp.∇obj(p.h))
	@info solve(ui, :semi_smooth_newton, linesearch = WolfePowell(), callback = calculate_error_uc!(1))
	@info solve(ui, :subgradientdescent, linesearch = BackTracking(), callback = calculate_error_uc!(2))
	@info solve(ui, :proximal_gradient, linesearch = WolfePowell(), callback = calculate_error_uc!(3))
	@info solve(ci, :newton, linesearch = WolfePowell(), callback = calculate_error_c!(4))
	@info solve(ci, :semi_smooth_newton, linesearch = Static(), callback = calculate_error_c!(5))

	for i in 1:size(data, 1)
		j = findfirst(x -> x < 1e-10, data[i, :])
		if j !== nothing && j < size(data, 2)
			data[i, j+1:end] .= -Inf
		end
	end
	data[data .== 0.0] .= 10e-14
	data
end
mp = 1 * ones(3)
p = Params2(10, 10 * ones(3), mp)
data1 = test_problem(test_fun_1_builder(200, 60), test_fun_1_conjugate_builder(200, 60), p)
fig1 = plot_convergencerate(data1)
data2 = test_problem(test_fun_2_builder(200, 60), test_fun_2_conjugate_builder(200, 60), p)
fig2 = plot_convergencerate(data2)


mp = [0.0, 0.0, 0.0]
p = Params2(16, 50 * [1.0, 1.0, 0.0], mp)
data3 = test_problem(test_fun_1_builder(80,65), test_fun_1_conjugate_builder(80, 65), p)
fig3 = plot_convergencerate(data3)

mp = 1e6 * [1.0, 0.0, 0.0]
p = Params2(71, 500 * [1.0, 0.0, 0.0], mp)
data4 = test_problem(test_fun_2_builder(1.23e6, 38), test_fun_2_conjugate_builder(1.23e6, 38), p)
fig4 = plot_convergencerate(data4)

p = Params2(70, 10 * ones(3), zeros(3))
data3 = [Vector{Float64}(undef, 0) for i in 1:5]
begin
	A = Matrix(I, (3, 3))
	b = 100 * ones(3)
	c = 0
	data3 = test_problem(quadratic_builder(A, b, c), quadratic_builder(A, b, c), p)
end

fig5 = plot_convergencerate(data3)


save("plots/cr1_lowh.png", fig1)
save("plots/cr2_lowh.png", fig2)
save("plots/cr1_highh.png", fig3)
save("plots/cr2_highh.png", fig4)
save("plots/cr_quadr.png", fig5)

#===========================================#
#Hysteresis currve

#Custom LineSearche


ms = 1.23 * 10^6
A = 38


Man(h) = 2 * ms / pi * atan(norm(h) / A)
## original problem

f = Figure()
ax = Axis(f[1, 1], xlabel = L"h_x", ylabel = L"m_x")
mp = [0.0, 0.0]
sol_prev = [0.0, 0.0]
sols = zeros(1000, 2)
for amount in [600, 180]

	hs = [[amount * sin(t), 0.0] for t in range(0, 4 * π, 1000)]
	for (i, h) in enumerate(hs)
		p = Params2(71, h, mp)
		problem = ConstrainedProblem(p.χ, h, p.mₚ, test_fun_2_conjugate_builder(ms, A))
		interface = Interface(problem, h, 20, 1e-14)
		sol = solve(interface, :newton, linesearch = WolfePowell())
		if !is_interior_solution(sol, interface)
			sol_prev = sol.sol
			mp = Man(sol.sol) * sol.sol / norm(sol.sol)
		end
		sols[i, :] .= mp
	end

	lines!(ax, stack(hs)[1, :], sols[1:end, 1], linestyle = amount == 180 ? :dash : :solid, color = amount == 180 ? :red : :blue, label = amount == 180 ? L"180 A/m" : L"600 A/m")

end
axislegend(ax, position = :lt)
f
save("plots/hysteresis_curve.png", f)

## dual problem
f = Figure()
ax = Axis(f[1, 1], xlabel = L"h_x", ylabel = L"m_x")
mp = [0.0, 0.0]
sol_prev = [0.0, 0.0]
sols = zeros(1000, 2)
for amount in [600, 180]

	hs = [[amount * sin(t), 0.0] for t in range(0, 4 * π, 1000)]
	for (i, h) in enumerate(hs)
		p = Params2(71, h, mp)
		problem = UnconstrainedProblem(p.χ, h, p.mₚ, test_fun_2_builder(ms, A))
		interface = Interface(problem, mp + [0.001, 1e-10], 10, 1e-14)
		sol = solve(interface, :semi_smooth_newton, linesearch = WolfePowell())
		mp = sol.sol
		sols[i, :] .= mp

	end

	lines!(ax, stack(hs)[1, :], sols[1:end, 1], linestyle = amount == 180 ? :dash : :solid, color = amount == 180 ? :red : :blue, label = amount == 180 ? L"180 A/m" : L"600 A/m")

end
axislegend(ax, position = :lt)

f
save("plots/hysteresis_curve_dual.png", f)

#2d
f = Figure()
ax = Axis(f[1, 1], xlabel = L"m_x", ylabel = L"m_y")
mp = [0.0, 0.0]
sol_prev = [0.0, 0.0]
sols = zeros(1000, 2)
Hm(t) = 110 * min(t / (6 * pi), 1)
hs = [Hm(t) * [cos(t), sin(t)] for t in range(0.0, 8 * π, 1000)]
for (i, h) in enumerate(hs)
	p = Params2(71, h, mp)
	problem = ConstrainedProblem(p.χ, h, p.mₚ, test_fun_2_conjugate_builder(ms, A))
	interface = Interface(problem, p.mₚ + [0.001, 1e-10], 10, 1e-14)
	sol = solve(interface, :semi_smooth_newton, linesearch = Static())
	if !is_interior_solution(sol, interface)
		sol_prev = sol.sol
		mp = Man(sol.sol) * sol.sol / norm(sol.sol)
	end
	mp = sol.sol
	sols[i, :] .= mp
end
lines!(ax, sols[1:end, 1], sols[1:end, 2])

f

save("plots/hysteresis_curve_2d.png", f)
f = Figure()
ax = Axis(f[1, 1], xlabel = L"m_x", ylabel = L"m_y")
mp = [0.0, 0.0]
sol_prev = [0.0, 0.0]
sols = zeros(1000, 2)
Hm(t) = 110 * min(t / (6 * pi), 1)
hs = [Hm(t) * [cos(t), sin(t)] for t in range(0.01, 8 * π, 1000)]
for (i, h) in enumerate(hs)
	p = Params2(71, h, mp)
	problem = UnconstrainedProblem(p.χ, h, p.mₚ, test_fun_2_builder(ms, A))
	interface = Interface(problem, mp + [0.001, 1e-10], 10, 1e-14)
	sol = solve(interface, :semi_smooth_newton, linesearch = WolfePowell())
	mp = sol.sol
	sols[i, :] .= mp
end
lines!(ax, sols[1:end, 1], sols[1:end, 2])

f
save("plots/hysteresis_curve_2d_dual.png", f)


