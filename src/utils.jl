include("interface.jl")
function transform_to_euklidean_3D(w, r, m)
	m + [r * sin(w[1]) * cos(w[2]), r * sin(w[1]) * sin(w[2]), r * cos(w[1])]
end


function transform_to_radial_3D(x, r, m)
	[acos((x[3] - m[3]) / r), atan(x[2] - m[2], x[1] - m[1])]
end

transform_to_euklidean_2D(w, r, m) = [r * cos(w[1]), r * sin(w[1])] + m

function transform_to_radial_2D(x, r, m)
	return [atan(x[2] - m[2], x[1] - m[1])]
end

function choleksyadaption!(A, β = 10e-3, max_iter = 1000)
	A = Symmetric(A)
	if (isposdef(A))
		return nothing
	end
	if any(diag(A) .<= 0)
		τ = 0
	else
		τ = minimum(diag(A)) - β
	end

	for i in 1:max_iter
		A = A + τ * I
		if isposdef(A)
			return nothing
		else
			τ = max(2 * τ, β)
		end
	end
	return nothing
end


short_circuit_exit(intf) = norm(intf.prob.∇U(intf.prob.mₚ) - intf.prob.h) ≤ intf.prob.χ

function checkconvergence!(cache::AbstractCache, intf::Interface)
	if isapprox(cache.xk, intf.prob.mₚ, atol = 10e-8)
		cache.err = norm(cache.xk - cache.xold)
    else
        cache.err = residium(cache.xk, intf)
    end
    
	if (cache.err ≤ intf.tol)
		return true
	end
    
    return false
end

residium(xk, intf) = norm(intf.prob.∇obj(xk))

function boundary_residium(xk::Vector{T}, intf::Interface)::T where {T <: Real}
	grad = intf.prob.∇obj(xk)
	n = xk - intf.prob.h
	norm(grad - ((grad ⋅ n) / (n ⋅ n)) * n)
end
