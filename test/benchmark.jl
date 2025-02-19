using BenchmarkTools
using Bachelorarbeit
using LineSearches
using GLMakie
include("testfunctions.jl")


mp = zeros(3)
h = 600 * ones(3)
χ = 71
cp = ConstrainedProblem(χ, h, mp, test_fun_2_builder(10e6, 38))
ci = Interface(cp, h + [0.1,0.1, 0.1], 100, 1e-6)
up = UnconstrainedProblem(χ, h, mp, test_fun_2_conjugate_builder(100, 30))
ui = Interface(up, mp , 1000, 1e-6)
sol = solve(ui, :proximal_gradient)
sol = solve(ui, :semi_smooth_newton)
sol = solve(ci, :semi_smooth_newton)
sol = solve(ci, :newton)

@benchmark solve($ci, :newton, linesearch = BackTracking())
@benchmark solve($ci, :semi_smooth_newton, linesearch = BackTracking())
@benchmark solve($ui, :semi_smooth_newton, linesearch = BackTracking())
@benchmark solve($ui, :subgradientdescent, linesearch = BackTracking())
@benchmark solve($ui, :proximal_gradient)

cp = ConstrainedProblem(χ, h, mp, test_fun_1_builder(100, 60))
up = UnconstrainedProblem(χ, h, mp, test_fun_1_conjugate_builder(100, 60))
ci = Interface(cp, h, 1000, 1e-6)
ui = Interface(up, mp + [0.0, 0.1, 0.1], 1000, 1e-6)

mp = zeros(3)
h = 50 * ones(3)
χ = 6
cp = ConstrainedProblem(χ, h, mp, test_fun_2_conjugate_builder(100, 60))
ci = Interface(cp, h, 100, 1e-6)
up = UnconstrainedProblem(χ, h, mp, test_fun_2_builder(100, 30))
ui = Interface(up, mp + [0.5, 0.5, 0.5], 1000, 1e-6)
sol = solve(ui, :proximal_gradient)
solve(ci, :newton)


@benchmark solve($ci, :newton, linesearch = BackTracking())
@benchmark solve($ci, :semi_smooth_newton, linesearch = BackTracking())
@benchmark solve($ui, :semi_smooth_newton, linesearch = BackTracking())
@benchmark solve($ui, :subgradientdescent, linesearch = BackTracking())
@benchmark solve($ui, :proximal_gradient, linesearch = BackTracking())
