include("interface.jl")

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

function subgradientdescent(intf::Interface{UnconstrainedProblem}; linesearch)
	cache = Cache(
		zeros(length(intf.x0)),
		intf.x0,
		intf.prob.obj(intf.x0),
		intf.prob.oracle(intf.x0),
		Inf,
	)

    ϕ(u) = intf.prob.obj(cache.xk + u * cache.s)
    dϕ(u) = intf.prob.oracle(cache.xk + u * cache.s) ⋅ cache.s
    ϕdϕ(u) = (ϕ(u), dϕ(u))
    for i in 1:intf.max_iter
        cache.s = - cache.dfk / norm(cache.dfk)
        α,cache.fk = linesearch(ϕ, dϕ, ϕdϕ, 1.0, cache.fk, dot(cache.dfk, cache.s)) 
        cache.xk = subgradientstep(cache, α)
        #@info cache.err, cache.s
        cache.fk = intf.prob.obj(cache.xk)
        cache.dfk = intf.prob.oracle(cache.xk)
        checkconvergence!(cache, intf) && return Solution(cache.xk, cache.fk, true, i, cache.err)
    end

    return Solution(cache.xk, cache.fk, false, intf.max_iter, cache.err)
end

subgradientstep(cache::Cache, α) = cache.xk + α * cache.s
