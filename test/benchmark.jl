using BenchmarkTools
using Bachelorarbeit
using LineSearches
include("testfunctions.jl")


suite = BenchmarkGroup()
suite["constrained"] = BenchmarkGroup()
suite["unconstrained"] = BenchmarkGroup()

mp = zeros(3)
h = 5*ones(3) 
χ = 1

cp = ConstrainedProblem(χ, h, mp, test_fun_2_conjugate_builder(100, 60))
up = UnconstrainedProblem(χ, h, mp, test_fun_2_builder(100, 60))
ci = Interface(cp, h, 1000, 1e-6)
ui = Interface(up, mp + [0.5, 0.5, 0.5], 50000, 1e-6)

@benchmark solve($ci, :newton, linesearch = BackTracking())
@benchmark solve($ci, :semi_smooth_newton, linesearch = BackTracking())
@benchmark solve($ui, :semi_smooth_newton, linesearch = BackTracking())
@benchmark solve($ui, :subgradientdescent, linesearch = BackTracking())
@benchmark solve($ui, :proximal_gradient)

sol = solve(ui, :subgradientdescent, linesearch = BackTracking())
sol = solve(ui, :proximal_gradient, linesearch = BackTracking())

cp = ConstrainedProblem(χ, h, mp, test_fun_1_builder(100, 60))
up = UnconstrainedProblem(χ, h, mp, test_fun_1_conjugate_builder(100, 60))
ci = Interface(cp, h, 1000, 1e-6)
ui = Interface(up, mp + [0.0, 0.1, 0.1], 1000, 1e-6)

@benchmark solve($ci, :newton, linesearch = BackTracking())
@benchmark solve($ci, :semi_smooth_newton, linesearch = BackTracking())
@benchmark solve($ui, :semi_smooth_newton, linesearch = BackTracking())
@benchmark solve($ui, :subgradientdescent, linesearch = BackTracking())
@benchmark solve($ui, :proximal_gradient, linesearch = BackTracking())

mp = zeros(3)
h = 70*ones(3) 
χ = 10

cp = ConstrainedProblem(χ, h, mp, test_fun_1_builder(80, 60))
up = UnconstrainedProblem(χ, h, mp, test_fun_1_conjugate_builder(80, 60))
ci = Interface(cp, h, 1000, 1e-6)
ui = Interface(up, mp + [0.0, 0.1, 0.1], 1000, 1e-6)
