struct ConstrainedProblem 
    χ::Float64
    h::Vector{Float64}
    mₚ::Vector{Float64}
    obj::Function
    ∇obj::Function
    ∇²obj::Function

    S::Function
    ∂S::Function
    ∂²S::Function
    
end

function ConstrainedProblem(χ, h, mₚ, S; kwags...)


    if !haskey(kwags, :jac)
        jac(x) = Zygote.gradient(S, x)[1]
    else
        jac = kwags[:jac]
    end

    if !haskey(kwags, :hes)
        hes(x) = Zygote.hessian(S, x)
    else
        hes = kwags[:hes]
    end

    obj(u) = S(u) - u ⋅ mₚ
    ∇obj(u) = jac(u) - mₚ
    ∇²obj = hes
    objᵩ(x) = obj(transform_to_euklidean_3D(x..., χ, h))
    ∇objᵩ(x) = Zygote.gradient(objᵩ, x)[1]
    ∇²objᵩ(x) = Zygote.hessian(objᵩ, x)
    ConstrainedProblem(χ, h, mₚ, obj, ∇obj, ∇²obj, S, jac, hes)
end

struct UnconstrainedProblem 
    χ::Float64
    h::Vector{Float64}
    mₚ::Vector{Float64}
    obj::Function
    ∇obj::Function
    ∇²obj::Function
    
    U::Function
    ∂U :: Function
    ∂²U :: Function

end

function UnconstrainedProblem(χ, h, mₚ, U; kwags...)

    if :jac ∉ keys(kwags)
        jac(x) = Zygote.gradient(U, x)[1]
    else
        jac = kwags[:jac]
    end


    if :hes ∉ keys(kwags)
        hes(x) = Zygote.hessian(U, x)
    else
        hes = kwags[:hes]
    end

    diffPart(m) = U(m) - h ⋅ m 
    ∇diffPart(m) = jac(m) - h

    obj(m) = U(m) - h ⋅ m + χ*norm(m - mₚ)
    ∇obj(m) = jac(m) - h + χ/norm(m - mₚ) * (m - mₚ)
    ∇²obj(m) = hes(m) + χ * (1/norm(m - mₚ) * I - 1/norm(m - mₚ)^3 * (m - mₚ) * (m - mₚ)')



    UnconstrainedProblem(χ, h, mₚ, obj,∇obj,∇²obj, U, jac, hes)


end