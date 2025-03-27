proxOfNorm(x, λ, mp) = ((1 - λ / max(norm(x - mp), λ)) * (x - mp)) + mp

function proxOfNorm!(res, x, λ, mp)
    norm_sq = 0.0
    @inbounds for i in eachindex(x, mp)
        norm_sq += (x[i] - mp[i])^2
    end
    norm_val = sqrt(norm_sq)
    
    # Compute the scaling factor
    scale = if norm_val > λ
        1 - λ / norm_val
    else
        0.0
    end
    
    # Update res in-place without temporary arrays
    @inbounds for i in eachindex(res, x, mp)
        res[i] = scale * (x[i] - mp[i]) + mp[i]
    end
    nothing
end

T(Lk, xk, dfk, χ, mₚ) = proxOfNorm((xk .- 1 / Lk * dfk), 1 / Lk * χ, mₚ)

T!(res, Lk, xk, dfk, χ, mₚ) = proxOfNorm!(res, (xk .- 1 / Lk * dfk), 1 / Lk * χ, mₚ)


function proximal_gradient(intf::Interface{UnconstrainedProblem}; callback)
	short_circuit_exit(intf) && return Solution(intf.prob.mₚ, intf.prob.obj(intf.prob.mₚ), true, 0, 0.0)

	s = 0.01
	η = 1.001
	Lk = s

	f(xk, intf) = intf.prob.U(xk) .- xk ⋅ intf.prob.h
	∇f(xk, intf) = intf.prob.∇U(xk) .- intf.prob.h

	cache = ProxGradCache(
		zeros(length(intf.x0)),
		copy(intf.x0),
		fill(Inf, length(intf.x0)),
		intf.prob.obj(intf.x0),
		f(intf.x0, intf),
		∇f(intf.x0, intf),
		T(Lk, intf.x0, ∇f(intf.x0, intf), intf.prob.χ, intf.prob.mₚ),
		norm(intf.prob.∇obj(intf.x0)),
		0)

	for cache.iter ∈ 1:intf.max_iter
		T!(cache.Tk, Lk, cache.xk, cache.dfk, intf.prob.χ, intf.prob.mₚ)
		while f(cache.Tk, intf) > cache.fk + dot(cache.dfk, (cache.Tk - cache.xk)) + Lk / 2 * norm(cache.Tk - cache.xk)^2
			Lk *= η
			T!(cache.Tk, Lk, cache.xk, cache.dfk, intf.prob.χ, intf.prob.mₚ)
		end
		cache.xold .= cache.xk
		cache.xk .= cache.Tk


		cache.gk = f(cache.xk, intf)
		cache.fk =	cache.gk +  intf.prob.χ * norm(cache.xk - intf.prob.mₚ)
		cache.dfk = ∇f(cache.xk, intf)
		converged = checkconvergence!(cache, intf)  

		if !isnothing(callback)
			callback(cache, intf)
		end
		if converged
			return Solution(cache.xk, cache.fk, true, cache.iter, cache.err)
		end
	end

	return Solution(cache.xk, intf.prob.obj(cache.xk), false, intf.max_iter, cache.err)
end
