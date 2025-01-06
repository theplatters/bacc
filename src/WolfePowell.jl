struct WolfePowell
	μ1::Float64
	σ::Float64
	τ::Float64
	τ1::Float64
	τ2::Float64
	ξ1::Float64
	ξ2::Float64
	max_iter::Int32
end

function WolfePowell()
	WolfePowell(0.1, 0.9, 0.2, 0.1, 0.6, 1, 10, 1000)
end

function (ls::WolfePowell)(ϕ, dϕ, ϕdϕ, α0, ϕ_0, dϕ_0)
	α_L = 0.0
	ϕ_L = ϕ(0.0)
	dϕ_L = dϕ(0.0)
	ϕ0 = ϕ(0)
	dϕ0 = dϕ(0)
	α_R = Inf64
	max_iter = 1000
	α_curr = α0
    ϕ_min = 1e-8
	for _ in 1:ls.max_iter
		ϕ_curr = ϕ(α_curr)
		if !isfinite(ϕ_curr)
			α_R, α_curr, = α_curr, α_L + ls.τ1 * (α_R - α_L)
		else
            if ϕ_curr < ϕ_min
                return α_curr,ϕ_curr
            end 
			if ϕ_curr > ϕ(0) + ls.μ1 * α_curr * dϕ0
				α_R = α_curr
				Δ = (α_R - α_L)
				c = (ϕ_curr - ϕ_L - dϕ_L * Δ) / (Δ^2.0)
				α_bar = α_L - dϕ_L / (2 * c)

				α_curr = min(max(α_L + ls.τ * Δ, α_bar), α_R - ls.τ * Δ)
			else
				dϕ_curr = dϕ(α_curr)
				if dϕ_curr < ls.σ * dϕ_0
					α_bar = 0.0
					if !isfinite(α_R)
						α_bar = (dϕ_L / dϕ_curr > (1 + ls.ξ2) / ls.ξ2) ?
								α_curr + (α_curr - α_L) * max(dϕ_curr / (dϕ_L - dϕ_curr), ls.ξ1) :
								α_curr + ls.ξ2*(α_curr - α_L)
					else
						α_bar = (dϕ_L / dϕ_curr > 1 + (α_curr - α_L) / (ls.τ2 * (α_R - α_curr))) ?
								α_curr + max((α_curr - α_L) * dϕ_curr / (dϕL - dϕ_curr), ls.τ1(α_R - α_curr)) :
								α_curr + ls.τ2 * (α_R - α_curr)
					end
					α_L = α_curr
					ϕ_L = ϕ_curr
					dϕ_L = dϕ_curr
					α_curr = α_bar
				else
					return α_curr, ϕ(α_curr)
				end
			end
		end
	end
	return α_curr, ϕ(α_curr)
end
