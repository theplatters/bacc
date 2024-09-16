using BenchmarkTools, Plots, Profile, Zygote, LinearAlgebra, LineSearches
using Bachelorarbeit
using CSV

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
        for method in [:newton, :semi_smooth_newton]
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

benchmark_results = run(suite)

CSV.write("data/benchmark_results.csv", benchmark_results) 

function simulate_hysteresis()#
    unit = 10e2
    mₛ = 1.23 * 10e6 * unit
    A = 38 * unit
    χ = 71 * unit
    f = test_fun_1_builder(mₛ, A)
    mₚ = zeros(2)
    steps = 100
    ms= zeros(2 * steps,2)
    h = ones(3)
    for i in 1:steps
        t = π/2 * i / steps
        h = [600 * unit * sin(t),0]
        intf = Interface(ConstrainedProblem(χ, h,mₚ, f), mₚ + unit * 10e-1 * ones(2), 2000, 1e-6)
        sol = solve(intf, :semi_smooth_newton, linesearch=StrongWolfe())
        @info sol.first
        mₚ = first(sol).sol
        ms[i,:] = mₚ
    end
    for i in 1:steps
        t = π/2 * (steps - i) / steps
        h = [600 * unit * sin(t),0]
        intf = Interface(ConstrainedProblem(χ, h,mₚ, f), mₚ - 10e-1 * unit * ones(2), 2000, 1e-6)
        sol = solve(intf, :semi_smooth_newton, linesearch=StrongWolfe())
        @info sol 
        mₚ = sol.sol
        @info norm(mₚ- h) - χ
        ms[steps + i,:] = mₚ
    end
    ms
end

m = simulate_hysteresis()
m[:,1] |> x -> Plots.plot(x, label="x")