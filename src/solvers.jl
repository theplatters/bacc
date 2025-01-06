using LinearSolve: LinearSolve
using LinearAlgebra
using LineSearches


include("utils.jl")
include("Solvers/newton.jl")
include("Solvers/subgradientdescent.jl")
include("Solvers/proximal_gradient.jl")
include("Solvers/semi_smooth_newton.jl")

function solve(intf::Interface, algorithm; linesearch = BackTracking(), callback::Union{Function, Nothing}=nothing)
    if algorithm == :proximal_gradient
        proximal_gradient(intf, callback = callback)
    elseif algorithm == :newton
        newton(intf, linesearch=linesearch, callback = callback)
    elseif algorithm == :semi_smooth_newton
        semi_smooth_newton(intf, linesearch=linesearch, callback = callback)
    elseif algorithm == :subgradientdescent
        subgradientdescent(intf, linesearch=linesearch, callback = callback)
    else
        throw(ArgumentError("Algorithm not implemented"))
    end
end


