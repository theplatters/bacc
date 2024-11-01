


struct SolutionComparison
	method::Symbol
	constrained::Bool
	accuracy::Float64
	fun::String
	converged::Bool
	iter::Int
end


suite = BenchmarkGroup()
suite["constrained"] = BenchmarkGroup()
suite["unconstrained"] = BenchmarkGroup()
iters_to_accuracy = Dict{String, Dict}()
iters_to_accuracy = Array{SolutionComparison, 1}()

quadr = (quadratic_builder(A, b, 0), quadratic_builder(A, b, 0))
fun1 = (test_fun_1_builder(ms, a), test_fun_1_conjugate_builder(ms, a))
fun2 = (test_fun_2_builder(ms2, a2), test_fun_2_conjugate_builder(ms2, a2))

for accuracy in [2.0^(-l) for l in 2:20]
	for f in ["quadr" => quadr, "fun1" => fun1, "fun2" => fun2]
		for method in [:proximal_gradient, :subgradientdescent, :semi_smooth_newton]
			unconstrained_prob = UnconstrainedProblem(χ, h, mₚ, f.second[1])
			unconstrained_intf = Interface(unconstrained_prob, mₚ + rand(3), 2000, accuracy)
			sol = solve(unconstrained_intf, method, linesearch = BackTracking())
			push!(iters_to_accuracy, SolutionComparison(method, false, accuracy, f.first, sol.converged, sol.iter))
			suite["constrained"][f.first][string(method)][accuracy] = @benchmarkable solve($unconstrained_intf, $method, linesearch = BackTracking())
		end
		for method in [:newton, :semi_smooth_newton]
			constrained_prob = ConstrainedProblem(χ, h, mₚ, f.second[2])
			const_intf = Interface(constrained_prob, mₚ + rand(3), 2000, accuracy)
			sol = solve(const_intf, method, linesearch = BackTracking())
			push!(iters_to_accuracy, SolutionComparison(method, true, accuracy, f.first, sol.converged, sol.iter))
			suite["unconstrained"][f.first][string(method)][accuracy] = @benchmarkable solve($const_intf, $method, linesearch = BackTracking())
		end
	end
end

function visualize(met, f, cons)
	res = map(x -> [x.accuracy, x.iter],
		filter(iters_to_accuracy) do sol_struct
			(; method, fun, converged, constrained) = sol_struct
			method == met && fun == f && converged && constrained == cons
		end)
	res
	f = Figure()
	ax = f[1, 1] = Axis(f)


	lines!(ax, [x[1] for x in res], [x[2] for x in res],
		label = string(met),
	)

	f
end
function visualize!(met, f, cons)
	res = map(
		filter(iters_to_accuracy) do sol_struct
			(; method, fun, converged, constrained) = sol_struct
			method == met && fun == f && converged && constrained == cons
		end) do x
		[x.accuracy, x.iter]
	end |> reverse
	Plots.plot!([(x[1]) for x in res], [x[2] for x in res],
		label = string(met),
		xlabel = "Accuracy",
		ylabel = "Iterations",
		title = "Iterations to reach accuracy",
		xscale = :log2)
end


iters_to_accuracy
visualize(:proximal_gradient, "quadr", false)
visualize!(:subgradientdescent, "quadr", false)
visualize!(:semi_smooth_newton, "quadr", false)

visualize(:proximal_gradient, "fun1", false)
visualize!(:subgradientdescent, "fun1", false)
visualize!(:semi_smooth_newton, "fun1", false)
visualize!(:newton, "fun1", true)
visualize!(:semi_smooth_newton, "fun1", true)


visualize(:proximal_gradient, "fun2", false)
visualize!(:subgradientdescent, "fun2", false)
visualize!(:semi_smooth_newton, "fun2", false)
visualize!(:newton, "fun2", true)
visualize!(:semi_smooth_newton, "fun2", true)


t2 = Threads.@spawn tune!(suite)

fetch(t2)


benchmark_results = run(suite)
benchmark_results["unconstrained"]["fun1"]["semi_smooth_newton"][0.00048828125] |> median
benchmark_results["unconstrained"]["fun1"]["newton"][0.00048828125] |> median
CSV.write("data/benchmark_results.csv", benchmark_results)