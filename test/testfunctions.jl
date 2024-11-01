function quadratic_builder(A::Matrix, b::Vector, c::Real)
    return x -> 0.5 * x' * A * x + b' * x
end

function quadratic_conjugate_builder(A::Matrix, b::Vector, c::Real)
    return x -> 0.5 * x' * A * x + b' * x
end
function test_fun_1_builder(mₛ, A)
    return x -> 2 / π * mₛ * (norm(x) * atan(norm(x) / A) + 1 / 2 * A * log(abs((norm(x) / A)^2 - 1)))
end

function test_fun_1_conjugate_builder(mₛ, A)
    return x -> 2 / π * mₛ * (norm(x) * atan(norm(x) / A) + 1 / 2 * A * log(abs((norm(x) / A)^2 - 1)))
end

function test_fun_2_builder(mₛ, A)
    return x -> 2 / π * mₛ * (norm(x) * atan(norm(x) / A) + 1 / 2 * A * log(abs((norm(x) / A)^2 - 1)))
end

function test_fun_2_conjugate_builder(mₛ, A)
    return x -> 2 / π * mₛ * (norm(x) * atan(norm(x) / A) + 1 / 2 * A * log(abs((norm(x) / A)^2 - 1)))
end
