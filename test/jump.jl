using JuMP, Ipopt

function solve_model(A::Matrix, b::Vector, h, chi, mp)
    m, n = size(A)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, hr[1:n])
    @constraint(model, sum((hr[i] - h[i])^2 for i in 1:n) <= chi^2)
    @objective(model, Min, 1/2 * hr' * A * hr + b' * hr - dot(hr, mp))
    optimize!(model)
    return value.(hr)
end

function solve_model2(A::Matrix, b::Vector, h, chi, mp)
    m, n = size(A)
    m, n = size(A)
    model = Model(Ipopt.Optimizer)
    invA = inv(A)
    set_silent(model)
    @variable(model, hr[1:n])
    @constraint(model, sum((hr[i] - h[i])^2 for i in 1:n) <= chi^2)
    @objective(model, Min, 1/2 * (hr-b)' * invA * (hr - b)- dot(hr, mp))
    optimize!(model)
    return value.(hr)
end

A = Matrix(I,3,3)
b = 100 * ones(3)

h = 10 * ones(3)
chi = 70 
mp = zeros(3)

solve_model(A,b,h,chi,mp)
solve_model2(A,b,h,chi,mp)