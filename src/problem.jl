struct ConstrainedProblem
	χ::Float64
	h::Vector{Float64}
	mₚ::Vector{Float64}
	obj::Function
	∇obj::Function
	∇²obj::Function

	S::Function
	∇S::Function
	∇²S::Function

end

function ConstrainedProblem(χ, h, mₚ, S; kwags...)


	if !haskey(kwags, :jac)
		jac(x) = Zygote.gradient(S, x)[1]
	else
		jac = kwags[:jac]
	end

	hes(x) = Zygote.hessian(S, x)

    obj(u) = S(u) - u ⋅ mₚ
    ∇obj(u) = jac(u) - mₚ
    ∇²obj = hes

    ConstrainedProblem(χ, h, mₚ, obj, ∇obj, ∇²obj, S, jac, hes)
end

struct UnconstrainedProblem
	χ::Float64
	h::Vector{Float64}
	mₚ::Vector{Float64}
	obj::Function
	∇obj::Function
	∇²obj::Function

	U::Function
	∇U::Function
	∇²U::Function

    oracle::Function
end

function UnconstrainedProblem(χ, h, mₚ, U; kwags...)

	if :jac ∉ keys(kwags)
		jac(x) = Zygote.gradient(U, x)[1]
	else
		jac = kwags[:jac]
	end


    hes(x) = Zygote.hessian(U, x)

	diffPart(m) = U(m) - h ⋅ m
	∇diffPart(m) = jac(m) - h

	obj(m) = U(m) - h ⋅ m + χ * norm(m - mₚ)
	∇obj(m) = jac(m) - h + χ / norm(m - mₚ) * (m - mₚ)
	∇²obj(m) = hes(m) + χ * Zygote.hessian(x -> norm(x - mₚ), m)
    oracle(m) = norm(m - mₚ) >= 1e-10 ? ∇obj(m) : ∇diffPart(m)

	UnconstrainedProblem(χ, h, mₚ, obj, ∇obj, ∇²obj, U, jac, hes, oracle)


end
