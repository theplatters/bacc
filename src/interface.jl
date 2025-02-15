include("problem.jl")
include("cache.jl")


struct Interface{T}
    prob::T
    x0::Vector{Float64}
    max_iter::Int64
    tol::Float64
end


struct Solution
    sol::Vector{Float64}
    f::Float64
    converged::Bool
    iter::Int64
    err::Float64
end

