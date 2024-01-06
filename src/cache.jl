mutable struct Cache
    xk::Vector{Float64}
    fk::Float64
    dfk::Vector{Float64}
    err::Float64
end
mutable struct NewtonCache
    s::Vector{Float64}
    xk::Vector{Float64}
    xold::Vector{Float64}
    fk::Float64
    fold::Float64
    dfk::Vector{Float64}
    Hfk::Matrix{Float64}
    err::Float64
    iter::Int64
end

