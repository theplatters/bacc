module Bachelorarbeit

using LinearAlgebra, Zygote
export UnconstrainedProblem,
	ConstrainedProblem,
	solve,
	Interface,
	proximal_gradient,
	transform_to_euklidean_3D,
	transform_to_radial_3D,
	transform_to_euklidean_2D,
	transform_to_radial_2D,
	WolfePowell,
	is_interior_solution
include("solvers.jl")
include("WolfePowell.jl")

is_interior_solution(sol::Solution, intf::Interface{ConstrainedProblem}) = norm(sol.sol - intf.prob.h) < intf.prob.χ
end # module Bachelorarbeit

