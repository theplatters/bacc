function semi_smooth_newton(intf::Interface{UnconstrainedProblem}; linesearch, callback)
	cache = NewtonCache(
		zeros(length(intf.x0)),
		intf.x0,
		intf.x0,
		intf.prob.obj(intf.x0),
		Inf,
		intf.prob.∇obj(intf.x0),
		intf.prob.∇²obj(intf.x0),
		Inf,
		0,
	)

	ϕ(α) = intf.prob.obj(cache.xk + α * cache.s)
	dϕ(α) = intf.prob.∇obj(cache.xk .+ α * cache.s) ⋅ cache.s
	ϕdϕ(α) = (ϕ(α), dϕ(α))


	for i ∈ 1:intf.max_iter

		if (!isnothing(callback))
			callback(cache, intf)
		end
		if cache.xk ≈ intf.prob.mₚ
			if norm(intf.prob.∂U(cache.xk) - intf.prob.h) ≤ intf.prob.χ
				return Solution(cache.xk, cache.fk, true, cache.iter, cache.err)
			else
				cache.xk = cache.xk + intf.prob.χ * (intf.prob.mₚ + rand(length(intf.x0)))
				cache.dfk = intf.prob.∇obj(cache.xk)
				cache.Hfk = intf.prob.∇²obj(cache.xk)
			end
		else
			cache.iter += 1

			newton_step!(cache)
			dϕ₀ = dot(cache.s, cache.dfk)

			α, cache.fk = linesearch(ϕ, dϕ, ϕdϕ, 1.0, cache.fk, dϕ₀)

			cache.xk += α * cache.s
			cache.dfk = intf.prob.∇obj(cache.xk)
			cache.Hfk = intf.prob.∇²obj(cache.xk)

			checkconvergence!(cache, intf) && return Solution(cache.xk, cache.fk, true, i, cache.err)

		end

	end
	return Solution(cache.xk, cache.fk, false, cache.iter, cache.err)
end


function semi_smooth_newton(intf::Interface{ConstrainedProblem}; linesearch, callback)

	g(x) = vcat(
		intf.prob.∇obj(x[1:end-1]) + 2 * x[end] * (x[1:end-1] - intf.prob.h),
		max(0, x[end] + norm(x[1:end-1] - intf.prob.h)^2 - intf.prob.χ^2) - x[end])

	function G(x)
		df1 = [
			(x[end] + norm(x[1:end-1] - intf.prob.h)^2 - intf.prob.χ^2 .> 0) .* 2 * (x[1:end-1] - intf.prob.h)
			(norm(x[1:end-1] - intf.prob.h)^2 - intf.prob.χ^2) + x[end] ≤ 0 ? -1 : 0]
		[intf.prob.∇²obj(x[1:end-1])+2*x[end]*I 2*(x[1:end-1]-intf.prob.h); df1']
	end

	x0 = vcat(intf.x0, 1)

	cache = NewtonCache(
		zeros(length(intf.x0) + 1),
		x0,
		x0,
		intf.prob.obj(intf.x0),
		Inf,
		g(x0),
		G(x0),
		norm(g(x0)),
		0,
	)
	ϕ(α) = intf.prob.obj(cache.xk[1:end-1] .+ α * cache.s[1:end-1])
	dϕ(α) = intf.prob.∇obj(cache.xk[1:end-1] .+ α * cache.s[1:end-1]) ⋅ cache.s[1:end-1]
	ϕdϕ(α) = (ϕ(α), dϕ(α))

	for i in 1:intf.max_iter

		if (!isnothing(callback))
			callback(cache, intf)
		end

		if (cache.err < intf.tol)
			return Solution(cache.xk[1:end-1], cache.fk, true, cache.iter, cache.err)
		end
		newton_step!(cache, false)
		println(cache.s)
		#@info cache.iter cache.s cache.Hfk
		dϕ₀ = dot(cache.s[1:end-1], cache.dfk[1:end-1])
		#α, cache.fk = linesearch(ϕ, dϕ, ϕdϕ, 1.0, cache.fk, dϕ₀)
		#println("Alpha:",α)
		α = 1
		cache.fk = intf.prob.obj(cache.xk[1:end-1] .+ α * cache.s[1:end-1])
		cache.iter += 1
		cache.xk = cache.xk .+ α * cache.s
		cache.dfk = g(cache.xk)
		cache.Hfk = G(cache.xk)
		cache.err = max(abs(cache.fk - cache.fold), maximum(abs.(cache.dfk)))
	end
	return (Solution(cache.xk[1:end-1], cache.fk, false, cache.iter, cache.err), cache.xk[end])
end
