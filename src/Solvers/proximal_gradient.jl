

function proxOfNorm(x, λ, mp)
    ((1 - λ / max(norm(x - mp), λ)) * (x - mp)) + mp
end


function proximal_gradient(intf::Interface{UnconstrainedProblem}; callback)
    s = 0.1
    η = 1.001
    Lk = 0.1

    f(xk) = intf.prob.U(xk) - xk ⋅ intf.prob.h
    ∂f(xk) = intf.prob.∂U(xk) - intf.prob.h
    cache = Cache(zeros(length(intf.x0)), intf.x0, f(intf.x0), ∂f(intf.x0), Inf)
    T(∂f, Lk, xk, dfk) = proxOfNorm((xk .- 1 / Lk * dfk), 1 / Lk * intf.prob.χ, intf.prob.mₚ)

    for i ∈ 1:intf.max_iter

        if(!isnothing(callback))
            callback(cache, intf)
        end

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