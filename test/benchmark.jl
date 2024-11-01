using BenchmarkTools 
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

