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
	fig = Figure(size=(1980,1020))
	ax1 = Axis(fig[1, 1], yscale =log10)
	ax2 = Axis(fig[1, 2], yscale =log10)
	linkyaxes!(ax1, ax2)
	for (i,l) in enumerate(["Semi-smooth Newton (unconstrained)", "Subgradient descent", "Proximal gradient"])
		lines!(ax1, data[i], label=l)
	end
	for (i,l) in enumerate(["Newton", "Semi-smooth Newton (constrained)"])
		lines!(ax2, data[i+3], label=l)
	end
	axislegend(ax1, "Methods", position = :lb)
	axislegend(ax2, "Methods", position = :lb)
	fig
end

function test_problem(fun, fun_conv, p::Params2)

	max_iter = 10
	tries = 1000
	data = [fill(1e-24,max_iter * tries) for _ in 1:5]
		function calculate_error!(i,j) 
			function a(cache, _) 
				data[i][max_iter *(j -1)  + (cache.iter)] = cache.err 
			end
		end


	for i in 1:tries
		up = UnconstrainedProblem(p.χ, p.h + 10 * rand(3), p.mₚ, fun)
		ui = Interface(up, p.mₚ + rand(3), max_iter, 1e-24)
		cp = ConstrainedProblem(p.χ, p.h, p.mₚ, fun_conv)
		ci = Interface(cp, p.h + rand(3), max_iter, 1e-24)

		solve(ui, :semi_smooth_newton, linesearch = BackTracking(), callback = calculate_error!(1,i))
		solve(ui, :subgradientdescent, linesearch = BackTracking(), callback = calculate_error!(2,i))
		solve(ui, :proximal_gradient, linesearch = BackTracking(), callback = calculate_error!(3,i))

		solve(ci, :newton, linesearch = BackTracking(), callback = calculate_error!(4,i))
		solve(ci, :semi_smooth_newton, linesearch = BackTracking(), callback = calculate_error!(5,i))
	end
	avg = [Matrix{Float64}(undef, 10, 10) for i in 1:5]
	map!(x -> reshape(x,max_iter,:), avg,data)
	map(x -> vec(mean(x,dims=2)),avg)
end

mp = zeros(3)
p = Params2(10,zeros(3),mp)

data1 = test_problem(test_fun_1_builder(60, 80), test_fun_1_conjugate_builder(60, 80), p)
fig1 = plot_convergencerate(data1)
save("plots/cr1.png", fig1)

data2 = test_problem(test_fun_2_builder(100, 60), test_fun_2_conjugate_builder(100, 60), p)
fig2 = plot_convergencerate(data2)
save("plots/cr2.png", fig2)

data3 = [Vector{Float64}(undef, 0) for i in 1:5]
begin
	A = 10 * randn(3, 3)
	A = A' * A
	A = (A + A') / 2
	b = rand(3)
	c = 5
	data3 = test_problem(quadratic_builder(A, b, c), quadratic_conjugate_builder(A, b, c), p)
end

fig3 = plot_convergencerate(data3)
save("plots/cr_quadr.png", fig3)