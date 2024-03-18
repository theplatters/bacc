using Revise
using Bachelorarbeit

using LinearAlgebra, Plots, Random, LineSearches


function M(x,A,mₛ)
    2 * mₛ * (x* atan(t/A) + 1/2 * A * Log[A^2] - 1/2 Log[A^2 + x^2]) / π
end

S(u) = sum(M(x, 10, 1e4),u)
