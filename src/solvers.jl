include("subgradientdescent.jl")
using LinearSolve: LinearSolve
using LinearAlgebra

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

function solve(intf::Interface, algorithm; linesearch)
    if algorithm == :proximal_gradient
        proximal_gradient(intf)
    elseif algorithm == :newton
        newton(intf, linesearch=linesearch)
    elseif algorithm == :semi_smooth_newton
        semi_smooth_newton(intf, linesearch=linesearch)
    elseif algorithm == :subgradientdescent
        subgradientdescent(intf, linesearch=linesearch)
    else
        throw(ArgumentError("Algorithm not implemented"))
    end
end



function newton(intf::Interface; linesearch)

    cache = NewtonCache(
        zeros(length(intf.x0)),
        intf.x0,
        intf.x0,
        intf.prob.obj(intf.x0),
        Inf,
        intf.prob.∇obj(intf.x0),
        intf.prob.∇²obj(intf.x0),
        Inf,
        0)

    ϕ(α) = intf.prob.obj(cache.xk .+ α * cache.s)
    dϕ(α) = intf.prob.∇obj(cache.xk .+ α * cache.s) ⋅ cache.s
    ϕdϕ(α) = (ϕ(α), dϕ(α))

    state = newton!(cache,
        linesearch,
        intf.prob.obj,
        intf.prob.∇obj,
        intf.prob.∇²obj,
        ϕ,
        dϕ,
        ϕdϕ,
        intf.max_iter,
        intf.tol,
        interior_violated=x -> norm(x - intf.prob.h) > intf.prob.χ)

    if state == :converged
        return Solution(cache.xk, cache.fk, true, cache.iter, cache.err)
    elseif state == :diverged
        return Solution(cache.xk, cache.fk, false, cache.iter, cache.err)
    else
        x0 = (intf.prob.χ / norm(cache.xk) * cache.xk) + intf.prob.h
        if length(cache.xk) == 2
            return newton_on_ball(intf, linesearch, x0, transform_to_euklidean_2D, transform_to_radial_2D, cache.iter)
        else
            return newton_on_ball(intf, linesearch, x0, transform_to_euklidean_3D, transform_to_radial_3D, cache.iter)
        end
    end

end

function newton_on_ball(intf::Interface, linesearch, x0, transform, inverse_transform, used_iter)

    f(x) = intf.prob.obj(transform(x, intf.prob.χ, intf.prob.h))
    g(x) = Zygote.gradient(u -> f(u), x)[1]
    H(x) = Zygote.hessian(u -> f(u), x)

    cache = NewtonCache(
        zeros(length(x0) - 1),
        inverse_transform(x0, intf.prob.χ, intf.prob.h),
        zeros(length(x0) - 1),
        intf.prob.obj(x0),
        Inf,
        g(inverse_transform(x0, intf.prob.χ, intf.prob.h)),
        H(inverse_transform(x0, intf.prob.χ, intf.prob.h)),
        Inf,
        used_iter
    )
    ϕ(α) = f(cache.xk + α * cache.s)
    dϕ(α) = g(cache.xk .+ α * cache.s) ⋅ cache.s
    ϕdϕ(α) = (ϕ(α), dϕ(α))

    state = newton!(cache,
        linesearch,
        f,
        g,
        H,
        ϕ,
        dϕ,
        ϕdϕ,
        intf.max_iter,
        intf.tol,
        guaranteedconvex=false)

    xk = transform(cache.xk, intf.prob.χ, intf.prob.h)
    Solution(xk, cache.fk, state == :converged ? true : false, cache.iter, cache.err)
end

function newton!(cache::NewtonCache, linesearch, f, g, H, ϕ, dϕ, ϕdϕ, max_iter, tol; interior_violated=(x -> false), guaranteedconvex=true)
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

        cache.err = max(abs(cache.fk - cache.fold), maximum(abs.(cache.dfk)))
        if cache.err <= tol
            return :converged
        end
    end

    return :diverged
end

function choleksyadaption!(A, β=10e-3, max_iter=1000)
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

function newton_step!(cache::NewtonCache, guaranteedconvex=true)
    cache.xold = copy(cache.xk)
    cache.fold = copy(cache.fk)
    if (!guaranteedconvex)
        choleksyadaption!(cache.Hfk)
    end
    prob = LinearSolve.LinearProblem(cache.Hfk, -cache.dfk)
    cache.s = LinearSolve.solve(prob).u
end


function proxOfNorm(x, λ, mp)
    ((1 - λ / max(norm(x - mp), λ)) * (x - mp)) + mp
end


function proximal_gradient(intf::Interface{UnconstrainedProblem})
    s = 0.1
    η = 1.001
    Lk = 0.1

    f(xk) = intf.prob.U(xk) - xk ⋅ intf.prob.h
    ∂f(xk) = intf.prob.∂U(xk) - intf.prob.h
    cache = Cache(zeros(length(intf.x0)), intf.x0, f(intf.x0), ∂f(intf.x0), Inf)
    T(∂f, Lk, xk, dfk) = proxOfNorm((xk .- 1 / Lk * dfk), 1 / Lk * intf.prob.χ, intf.prob.mₚ)

    for i ∈ 1:intf.max_iter

        while f(T(∂f, Lk, cache.xk, cache.dfk)) > cache.fk + dot(cache.dfk, (T(∂f, Lk, cache.xk, cache.dfk) - cache.xk)) + Lk / 2 * norm(T(∂f, Lk, cache.xk, cache.dfk) - cache.xk)^2
            Lk = Lk * η
        end

        cache.xk = T(∂f, Lk, cache.xk, cache.dfk)
        cache.fk = f(cache.xk)
        cache.dfk = ∂f(cache.xk)

        checkconvergence!(cache, intf) && return Solution(cache.xk, cache.fk + intf.prob.χ * norm(cache.xk - intf.prob.mₚ), true, i, cache.err)
    end

    return Solution(cache.xk, intf.prob.obj(cache.xk), false, intf.max_iter, cache.err)
end

function semi_smooth_newton(intf::Interface{UnconstrainedProblem}; linesearch)
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


function semi_smooth_newton(intf::Interface{ConstrainedProblem}; linesearch)

    g(x) = vcat(
        intf.prob.∇obj(x[1:end-1]) + 2 * x[end] * x[1:end-1],
        max(0, x[end] + norm(x[1:end-1] - intf.prob.h)^2 - intf.prob.χ^2) - x[end])

    function G(x)
        df1 = [
            (x[end] + norm(x[1:end-1] - intf.prob.h)^2 - intf.prob.χ^2 .> 0) .* 2 * (x[1:end-1] - intf.prob.h)
            (norm(x[1:end-1] - intf.prob.h)^2 - intf.prob.χ^2) + x[end] ≤ 0 ? -1 : 0]
        [intf.prob.∇²obj(x[1:end-1])+2*x[end]*I 2*x[1:end-1]-intf.prob.h; df1']
    end

    x0 = vcat(intf.x0, intf.prob.χ - norm(intf.x0 - intf.prob.h))

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
        if (cache.err < intf.tol)
            return Solution(cache.xk[1:end-1], cache.fk, true, cache.iter, cache.err)
        end
        newton_step!(cache, false)
		println(cache.s)
        @info cache.iter cache.s cache.Hfk
        dϕ₀ = dot(cache.s[1:end-1], cache.dfk[1:end-1])
        α, cache.fk = linesearch(ϕ, dϕ, ϕdϕ, 1.0, cache.fk, dϕ₀)
		
		cache.iter += 1
  		cache.xk = cache.xk .+ α * cache.s
        cache.dfk = g(cache.xk)
        cache.Hfk = G(cache.xk)
        cache.err = max(abs(cache.fk - cache.fold), maximum(abs.(cache.dfk)))
    end
    return (Solution(cache.xk[1:end-1], cache.fk, false, cache.iter, cache.err))
end