function quadratic_builder(A::Matrix, b::Vector, c::Real)
	return x -> 0.5 * x' * A * x + b' * x
end

function quadratic_conjugate_builder(A::Matrix, b::Vector, c::Real)
	return x -> 0.5 * (x - b)' * inv(A) * (x - b) - c
end

function test_fun_1_builder(mₛ, a)
	return x -> a * (norm(x) * atanh(norm(x) / mₛ) + 1 / 2 * mₛ * (log(norm(m)^2 + mₛ^2) - 2 * log(mₛ)))
end

function test_fun_1_conjugate_builder(mₛ, a)
	return x -> a * mₛ * log(cosh(norm(x) / a))
end

function test_fun_2_builder(mₛ, A)
	return x -> 2 * mₛ / π * (norm(x) * atan(norm(x) / A) - 1 / 2 * log((norm(x) / A)^2 + 1))
end

function test_fun_2_conjugate_builder(mₛ, A)
	return x -> (2 * A * mₛ / π) * log(sec(π / (2 * mₛ) * norm(x)))
end
