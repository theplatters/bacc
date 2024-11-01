
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

function checkconvergence!(cache::AbstractCache, intf::Interface)
	if isapprox(cache.xk,intf.prob.mₚ,rtol = intf.tol)
        cache.err = norm(cache.s)
		if norm(intf.prob.∂U(intf.prob.mₚ) - intf.prob.h) ≤ intf.prob.χ
			return true
		end
	else
		cache.err = norm(intf.prob.∇obj(cache.xk))
		if (cache.err ≤ intf.tol)
			return true
		end
	end
	false
end

function residium(cache::T, intf) where {T <: AbstractCache}
    
end