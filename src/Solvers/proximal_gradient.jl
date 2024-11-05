

function proxOfNorm(x, λ, mp)
	((1 - λ / max(norm(x - mp), λ)) * (x - mp)) + mp
end


function proximal_gradient(intf::Interface{UnconstrainedProblem}; callback)
	short_circuit_exit(intf) && return Solution(intf.prob.mₚ, intf.prob.obj(intf.prob.mₚ), true, 0, 0.0)
	s = 0.1
	η = 1.0001
	Lk = 0.01

	T(Lk, xk, dfk) = proxOfNorm((xk .- 1 / Lk * dfk), 1 / Lk * intf.prob.χ, intf.prob.mₚ)
	f(xk) = intf.prob.U(xk) - xk ⋅ intf.prob.h
	∇f(xk) = intf.prob.∇U(xk) - intf.prob.h


	cache = ProxGradCache(
		zeros(length(intf.x0)),
		intf.x0, f(intf.x0),
		∇f(intf.x0),
		T(Lk, intf.x0, ∇f(intf.x0)),
		norm(intf.prob.∇obj(intf.x0)),
		0)

	for cache.iter ∈ 1:intf.max_iter
		if !isnothing(callback)
			callback(cache, intf)
		end

		while f(cache.Tk) > cache.fk + dot(cache.dfk, (cache.Tk - cache.xk)) + Lk / 2 * norm(cache.Tk - cache.xk)^2
			Lk *= η
			cache.Tk = T(Lk, cache.xk, cache.dfk)
		end

		cache.xk = copy(cache.Tk)
		cache.fk = f(cache.xk)
		cache.dfk = ∇f(cache.xk)
		checkconvergence!(cache, intf) && return Solution(cache.xk, cache.fk + intf.prob.χ * norm(cache.xk - intf.prob.mₚ), true, cache.iter, cache.err)
	end

	return Solution(cache.xk, intf.prob.obj(cache.xk), false, intf.max_iter, cache.err)
end
