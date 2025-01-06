
function newton(intf::Interface; linesearch, callback::Union{Nothing, Function} = nothing)

	if norm(intf.x0 - intf.prob.h) > intf.prob.χ
		error("The initial point is not on the ball.")
	end


	cache = NewtonCache(
		zeros(length(intf.x0)),
		intf.x0,
		fill(typemax(eltype(intf.x0)), length(intf.x0)),
		intf.prob.obj(intf.x0),
		Inf,
		intf.prob.∇obj(intf.x0),
		intf.prob.∇²obj(intf.x0),
		norm(intf.prob.∇obj(intf.x0)),
		0)

	ϕ(α) = intf.prob.obj(cache.xk .+ α * cache.s)
	dϕ(α) = intf.prob.∇obj(cache.xk .+ α * cache.s) ⋅ cache.s
	ϕdϕ(α) = (ϕ(α), dϕ(α))

	state = newton!(
		intf,
		cache,
		linesearch,
		intf.prob.obj,
		intf.prob.∇obj,
		intf.prob.∇²obj,
		ϕ,
		dϕ,
		ϕdϕ,
		intf.max_iter,
		intf.tol,
		interior_violated = x -> norm(x - intf.prob.h) > intf.prob.χ,
		callback = callback,
		error_function = (cache,intf) -> residium(cache.xk,intf))

	if state == :converged
		return Solution(cache.xk, cache.fk, true, cache.iter, cache.err)
	elseif state == :diverged
		return Solution(cache.xk, cache.fk, false, cache.iter, cache.err)
	else
		x0 = (intf.prob.χ / norm(cache.xk) * cache.xk) + intf.prob.h
		if length(cache.xk) == 2
			return newton_on_ball(intf, linesearch, x0, transform_to_euklidean_2D, transform_to_radial_2D, cache.iter, callback = callback)
		elseif length(cache.xk) == 3
			return newton_on_ball(intf, linesearch, x0, transform_to_euklidean_3D, transform_to_radial_3D, cache.iter, callback = callback)
		else
			error("This method is not implemented for problems, that are not 2D or 3D.")
		end
	end

end

function newton_on_ball(intf::Interface, linesearch, x0, transform, inverse_transform, used_iter; callback::Union{Nothing, Function} = nothing)

	f(x) = intf.prob.obj(transform(x, intf.prob.χ, intf.prob.h))
	g(x) = Zygote.gradient(u -> f(u), x)[1]
	H(x) = Zygote.hessian(u -> f(u), x)

	cache = NewtonCache(
		zeros(length(intf.x0) - 1),
		inverse_transform(x0, intf.prob.χ, intf.prob.h),
		fill(typemax(eltype(x0)), length(x0) - 1),
		intf.prob.obj(x0),
		Inf,
		g(inverse_transform(x0, intf.prob.χ, intf.prob.h)),
		H(inverse_transform(x0, intf.prob.χ, intf.prob.h)),
		norm(intf.prob.∇obj(x0)),
		used_iter,
	)
	ϕ(α) = f(cache.xk + α * cache.s)
	dϕ(α) = g(cache.xk .+ α * cache.s) ⋅ cache.s
	ϕdϕ(α) = (ϕ(α), dϕ(α))

	if length(cache.xk) == 1
		err_fun = (cache, intf) -> boundary_residium(transform_to_euklidean_2D(cache.xk, intf.prob.χ, intf.prob.h), intf)
	else
		err_fun = (cache, intf) -> boundary_residium(transform_to_euklidean_3D(cache.xk, intf.prob.χ, intf.prob.h), intf)
	end

	state = newton!(
		intf,
		cache,
		linesearch,
		f,
		g,
		H,
		ϕ,
		dϕ,
		ϕdϕ,
		intf.max_iter,
		intf.tol,
		guaranteedconvex = false,
		callback = callback,
		error_function = err_fun)

	xk = transform(cache.xk, intf.prob.χ, intf.prob.h)
	Solution(xk, cache.fk, state == :converged ? true : false, cache.iter, cache.err)
end

function newton!(
	intf::Interface,
	cache::NewtonCache,
	linesearch,
	f,
	g,
	H,
	ϕ,
	dϕ,
	ϕdϕ,
	max_iter,
	tol;
	interior_violated = (x -> false),
	guaranteedconvex = true,
	callback::Union{Nothing, Function} = nothing,
	error_function = (cache, intf) -> max(abs(cache.fk - cache.fold), maximum(abs.(cache.dfk))),
	)
	for cache.iter ∈ 1:max_iter
		newton_step!(cache, guaranteedconvex)
		dϕ₀ = dot(cache.s, cache.dfk)
		α, cache.fk = linesearch(ϕ, dϕ, ϕdϕ, 1.0, cache.fk, dϕ₀)
		
		cache.xk += α * cache.s
		if interior_violated(cache.xk)
			return :outside
		end
		
		cache.dfk = g(cache.xk)
		cache.Hfk = H(cache.xk)
		
		cache.err = error_function(cache, intf)
		if !isnothing(callback)
			callback(cache, intf)
		end
		if cache.err <= tol
			return :converged
		end
	end

	return :diverged
end

function newton_step!(cache::AbstractCache, guaranteedconvex = true)
	cache.xold = copy(cache.xk)
	cache.fold = copy(cache.fk)
	if (!guaranteedconvex)
		choleksyadaption!(cache.Hfk)
	end
	prob = LinearSolve.LinearProblem(cache.Hfk, -cache.dfk)
	cache.s = LinearSolve.solve(prob).u
end
