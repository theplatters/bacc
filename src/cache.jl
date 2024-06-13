abstract type AbstractCache end
mutable struct Cache <: AbstractCache
    s::Vector{Float64}
    xk::Vector{Float64}
    fk::Float64
    dfk::Vector{Float64}
    err::Float64
end
mutable struct NewtonCache <: AbstractCache
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

function print(io::IO, cache::NewtonCache)
    println("On Iteratation $(cache.iter):")
    println("Function value f = $(cache.fk) with xk = $(cache.xk)")
end

function print(io::IO, cache::Cache)
    println("On Iteratation $(cache.iter):")
    println("Function value f = $(cache.fk) with xk = $(cache.xk)")
end

