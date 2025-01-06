
proxOfNorm(x, λ, mp) = ((1 - λ / max(norm(x - mp), λ)) * (x - mp)) + mp
function proxOfNorm!(res, x, λ, mp)
	res .= ((1 - λ / max(norm(x - mp), λ)) * (x - mp)) + mp
	nothing
end

T(Lk, xk, dfk, χ, mₚ) = proxOfNorm((xk .- 1 / Lk * dfk), 1 / Lk * χ, mₚ)

T!(res, Lk, xk, dfk, χ, mₚ) = proxOfNorm!(res, (xk .- 1 / Lk * dfk), 1 / Lk * χ, mₚ)


function proximal_gradient(intf::Interface{UnconstrainedProblem}; callback)
	short_circuit_exit(intf) && return Solution(intf.prob.mₚ, intf.prob.obj(intf.prob.mₚ), true, 0, 0.0)

	s = 0.01
	η = 1.001
	Lk = s

	f(xk, intf) = intf.prob.U(xk) - xk ⋅ intf.prob.h
	∇f(xk, intf) = intf.prob.∇U(xk) - intf.prob.h

	cache = ProxGradCache(
		zeros(length(intf.x0)),
		copy(intf.x0),
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
		cache.xk = copy(cache.Tk)
		cache.fk = f(cache.xk, intf)
		cache.dfk = ∇f(cache.xk, intf)
		converged = checkconvergence!(cache, intf)  

		if !isnothing(callback)
			callback(cache, intf)
		end
		if converged
			return Solution(cache.xk, cache.fk + intf.prob.χ * norm(cache.xk - intf.prob.mₚ), true, cache.iter, cache.err)
		end
	end

	return Solution(cache.xk, intf.prob.obj(cache.xk), false, intf.max_iter, cache.err)
end
