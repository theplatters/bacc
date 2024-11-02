function semi_smooth_newton(intf::Interface{UnconstrainedProblem}; linesearch, callback)

	short_circuit_exit(intf) && return Solution(intf.prob.mₚ, intf.prob.obj(intf.prob.mₚ), true, 0, 0.0)

	cache = NewtonCache(
		zeros(length(intf.x0)),
		intf.x0,
		fill(typemax(eltype(intf.x0)), length(intf.x0)),
		Inf,
		intf.prob.obj(intf.x0),
		intf.prob.∇obj(intf.x0),
		intf.prob.∇²obj(intf.x0),
		norm(intf.prob.∇obj(intf.x0)),
		0,
	)

	ϕ(α) = intf.prob.obj(cache.xk + α * cache.s)
	dϕ(α) = intf.prob.∇obj(cache.xk .+ α * cache.s) ⋅ cache.s
	ϕdϕ(α) = (ϕ(α), dϕ(α))

	for cache.iter ∈ 1:intf.max_iter	

		if !isnothing(callback)
			callback(cache, intf)
		end
		
		if cache.xk ≈ intf.prob.mₚ
            cache.dfk = intf.prob.∇U(cache.xk) - intf.prob.h + ones(length(intf.x0)) / sqrt(length(intf.x0))
            cache.Hfk = intf.prob.∇²U(cache.xk)
		else
            cache.dfk = intf.prob.∇obj(cache.xk)
            cache.Hfk = intf.prob.∇²obj(cache.xk)
        end
        
        newton_step!(cache)
		
        dϕ₀ = dot(cache.s, cache.dfk)
        
        α, cache.fk = linesearch(ϕ, dϕ, ϕdϕ, 1.0, cache.fk, dϕ₀)
        cache.xk += α * cache.s
        
        checkconvergence!(cache, intf) && return Solution(cache.xk, cache.fk, true, cache.iter, cache.err)

	end
	return Solution(cache.xk, cache.fk, false, cache.iter, cache.err)
end

function calculate_error!(cache, intf)
	if (isapprox(norm(cache.xk[1:end-1] - intf.prob.h) - intf.prob.χ, 0, atol = 1e-8))
		cache.err = boundary_residium(cache.xk[1:end-1], intf)
	else
		cache.err = residium(cache.xk[1:end-1], intf)
	end
    nothing
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
		norm(intf.prob.∇obj(intf.x0)),
		g(x0),
		G(x0),
		norm(intf.prob.∇obj(intf.x0)),
		0,
	)
	ϕ(α) = intf.prob.obj(cache.xk[1:end-1] .+ α * cache.s[1:end-1])
	dϕ(α) = intf.prob.∇obj(cache.xk[1:end-1] .+ α * cache.s[1:end-1]) ⋅ cache.s[1:end-1]
	ϕdϕ(α) = (ϕ(α), dϕ(α))
	
	for cache.iter in 1:intf.max_iter
		
		if (!isnothing(callback))
			callback(cache, intf)
		end
        
		if (cache.err < intf.tol)
			return Solution(cache.xk[1:end-1], cache.fk, true, cache.iter, cache.err)
		end
		newton_step!(cache, false)
		dϕ₀ = dot(cache.s[1:end-1], cache.dfk[1:end-1])
		α = 1
		cache.fk = intf.prob.obj(cache.xk[1:end-1] .+ α * cache.s[1:end-1])
		cache.xk = cache.xk .+ α * cache.s
		cache.dfk = g(cache.xk)
		cache.Hfk = G(cache.xk)
        
        calculate_error!(cache,intf)

	end
	return Solution(cache.xk[1:end-1], cache.fk, false, cache.iter, cache.err)
end
