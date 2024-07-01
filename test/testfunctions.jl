using BenchmarkTools, Plots, Profile, Zygote, LinearAlgebra, LineSearches
using Bachelorarbeit

function quadratic_builder(A::Matrix, b::Vector, c::Real)
    return x -> 0.5 * x' * A * x + b' * x
end

function test_fun_1_builder(mₛ, A)
    return x -> 2 / π * mₛ * (norm(x) * atan(norm(x) / A) + 1 / 2 * A * log(abs((norm(x) / A)^2 - 1)))
end

suite = BenchmarkGroup()
χ = 1.6
mₚ = rand(3)
A = diagm([5.0, 2.0, 1])
b = h = ones(3)
a = 60
ms = 8

suite["constrained"] = BenchmarkGroup()
suite["unconstrained"] = BenchmarkGroup()
iters_to_accuracy = Dict{String,Dict}()


struct SolutionComparison
    method::Symbol
    accuracy::Float64
    fun::String
    converged::Bool
    iter::Int
end

iters_to_accuracy = Array{SolutionComparison,1}()

for accuracy in [2.0^(-l) for l in 2:20]
    for f in ["quadr" => quadratic_builder(A, b, 0), "lit" => test_fun_1_builder(ms, a)]
        for method in [:proximal_gradient, :subgradientdescent, :semi_smooth_newton]
            unconstrained_prob = UnconstrainedProblem(χ, h, mₚ, f.second)
            unconstrained_intf = Interface(unconstrained_prob, -ones(3), 20000, accuracy)
            sol = solve(unconstrained_intf, method, linesearch=BackTracking())
            push!(iters_to_accuracy, SolutionComparison(method, accuracy, f.first, sol.converged, sol.iter))
            suite["constrained"][f.first][string(method)][accuracy] = @benchmarkable solve($unconstrained_intf, $method, linesearch=BackTracking())
        end
        for method in [:newton]
            constrained_prob = ConstrainedProblem(χ, h, mₚ, f.second)
            const_intf = Interface(constrained_prob, -ones(3), 2000, accuracy)
            sol = solve(const_intf, method, linesearch=BackTracking())
            push!(iters_to_accuracy, SolutionComparison(method, accuracy, f.first, sol.converged, sol.iter))
            suite["unconstrained"][f.first][string(method)][accuracy] = @benchmarkable solve($const_intf, $method, linesearch=BackTracking())
        end
    end
end

function visualize(met, f)
    res = map(x -> [x.accuracy, x.iter],
        filter(iters_to_accuracy) do x
            (; method, fun, converged) = x
            method == met && fun == f && converged
        end)
    Plots.plot([x[1] for x in res], [x[2] for x in res],
        label=string(met),
        xlabel="Accuracy",
        ylabel="Iterations",
        title="Iterations to reach accuracy",
        xscale=:log2)
end
function visualize!(met, f)
    res = map(filter(iters_to_accuracy) do x
        (; method, fun, converged) = x
        method == met && fun == f && converged
    end) do x
        [x.accuracy, x.iter]
    end |> reverse
    Plots.plot!([(x[1]) for x in res], [x[2] for x in res],
        label=string(met),
        xlabel="Accuracy",
        ylabel="Iterations",
        title="Iterations to reach accuracy",
        xscale=:log2)
end

visualize(:proximal_gradient, "quadr")
visualize!(:subgradientdescent, "quadr")
visualize!(:semi_smooth_newton, "quadr")

visualize(:proximal_gradient, "lit")
visualize!(:subgradientdescent, "lit")
visualize!(:semi_smooth_newton, "lit")

visualize(:newton, "lit")
visualize!(:newton, "quadr")


tune!(suite)

suite

benchmark_results = run(suite)

f = test_fun_1_builder(ms, a)
plotfun(x, y) = f([x, y])


x = range(-66, 66, length=100)
y = range(-66, 66, length=100)
Plots.surface(x, y, plotfun)

f([65, 0])