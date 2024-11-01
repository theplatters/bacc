using LinearSolve: LinearSolve
using LinearAlgebra


include("utils.jl")
include("Solvers/newton.jl")
include("Solvers/subgradientdescent.jl")
include("Solvers/proximal_gradient.jl")
include("Solvers/semi_smooth_newton.jl")

function solve(intf::Interface, algorithm; linesearch)
    if algorithm == :proximal_gradient
        proximal_gradient(intf)
    elseif algorithm == :newton
        newton(intf, linesearch=linesearch)
    elseif algorithm == :semi_smooth_newton
        semi_smooth_newton(intf, linesearch=linesearch)
    elseif algorithm == :subgradientdescent
        subgradientdescent(intf, linesearch=linesearch)
    else
        throw(ArgumentError("Algorithm not implemented"))
    end
end


