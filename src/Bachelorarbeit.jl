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
    WolfePowell
include("solvers.jl")
include("WolfePowell.jl")
end # module Bachelorarbeit
