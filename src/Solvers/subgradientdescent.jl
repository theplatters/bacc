function subgradientdescent(intf::Interface{UnconstrainedProblem}; linesearch, callback)

	cache = Cache(
		zeros(length(intf.x0)),
		intf.x0,
		intf.prob.obj(intf.x0),
		intf.prob.oracle(intf.x0),
		Inf,
	)

	short_circuit_exit(intf) && return Solution(intf.prob.mₚ, intf.prob.obj(intf.prob.mₚ), true, 0, 0.0)

	ϕ(u) = intf.prob.obj(cache.xk + u * cache.s)
	dϕ(u) = intf.prob.oracle(cache.xk + u * cache.s) ⋅ cache.s
	ϕdϕ(u) = (ϕ(u), dϕ(u))

	for i in 1:intf.max_iter
		cache.s = -cache.dfk
		α, cache.fk = linesearch(ϕ, dϕ, ϕdϕ, 4.0, cache.fk, dot(cache.dfk, cache.s))
		cache.xk += α * cache.s

		cache.fk = intf.prob.obj(cache.xk)
		cache.dfk = intf.prob.oracle(cache.xk)

		checkconvergence!(cache, intf) && return Solution(cache.xk, cache.fk, true, i, cache.err)
		if (!isnothing(callback))
			callback(cache, intf)
		end
	end

	return Solution(cache.xk, cache.fk, false, intf.max_iter, cache.err)
end
