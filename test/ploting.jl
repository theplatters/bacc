using Bachelorarbeit
using LineSearches
using LinearAlgebra
using GLMakie
using Random
using Statistics

include("testfunctions.jl")

struct Params2
  χ::Float64
  h::Vector{Float64}
  mₚ::Vector{Float64}
end

function plot_convergencerate(data)
  fig = Figure(size=(1980, 1020))
  ax1 = Axis(fig[1, 1],  xticks=1:maximum(length.(data)))
  ax2 = Axis(fig[1, 2],  xticks=1:maximum(length.(data)))
  for (l,d) in zip(["Semi-smooth Newton (unconstrained)", "Subgradient descent", "Proximal gradient"],eachrow(data[1:3,:]))
    lines!(ax1, d[d .!= -Inf], label=l)
  end
  for (l,d) in zip(["Newton", "Semi-smooth Newton (constrained)"],eachrow(data[4:5,:]))
    lines!(ax2, d[d .!= -Inf], label=l)
  end
  axislegend(ax1, "Methods", position=:lb)
  axislegend(ax2, "Methods", position=:lb)
  fig
end
import Base
function test_problem(fun, fun_conv, p::Params2)

  max_iter = 10
  data = fill(-Inf,5, max_iter + 1)
  function calculate_error!(i)
    function a(cache, _)
      data[i, 1 + cache.iter] = cache.err
    end
  end
  
  
  h = p.h
  up = UnconstrainedProblem(p.χ, h, p.mₚ, fun)
  ui = Interface(up, p.mₚ, max_iter, 1e-10)
  cp = ConstrainedProblem(p.χ, h, p.mₚ, fun_conv)
  ci = Interface(cp, p.h, max_iter, 1e-10)
  
  data[1:3,1] .=  norm(up.oracle(p.mₚ))
  data[4:5,1] .=  norm(cp.∇obj(p.h))

  @info solve(ui, :semi_smooth_newton, linesearch=MoreThuente(), callback=calculate_error!(1))
  @info sol = solve(ui, :subgradientdescent, linesearch=MoreThuente(), callback=calculate_error!(2))
  @info sol = solve(ui, :proximal_gradient, linesearch=MoreThuente(), callback=calculate_error!(3))
  @info solve(ci, :newton, linesearch=MoreThuente(), callback=calculate_error!(4))
  @info solve(ci, :semi_smooth_newton, linesearch=MoreThuente(), callback=calculate_error!(5))
  data
end
mp = 1 * ones(3)
p = Params2(10, 10 * ones(3), mp)
data1 = test_problem(test_fun_1_builder(200, 60), test_fun_1_conjugate_builder(200, 60), p)
fig1 = plot_convergencerate(data1)
data2 = test_problem(test_fun_2_builder(200, 60), test_fun_2_conjugate_builder(200, 60), p)
fig2 = plot_convergencerate(data2)


mp = 10 * ones(3)
p = Params2(50, 100 * ones(3), zeros(3))
data3 = test_problem(test_fun_1_builder(400, 60), test_fun_1_conjugate_builder(400, 60), p)
fig3 = plot_convergencerate(data3)
data4 = test_problem(test_fun_2_builder(400, 60), test_fun_2_conjugate_builder(400, 60), p)
fig4 = plot_convergencerate(data4)

p = Params2(10, 10 * ones(3), zeros(3))
data3 = [Vector{Float64}(undef, 0) for i in 1:5]
begin
  Q = 10 * rand(3, 3)
  A = Q * Diagonal([10.0, 1.0, 14.0]) * Q'
  b = 10 * rand(3)
  c = 5
  data3 = test_problem(quadratic_builder(A, b, c), quadratic_conjugate_builder(A, b, c), p)
end

fig5 = plot_convergencerate(data3)

save("plots/cr1_lowh.png", fig1)
save("plots/cr2_lowh.png", fig2)
save("plots/cr1_highh.png", fig3)
save("plots/cr2_highh.png", fig4)
save("plots/cr_quadr.png", fig5)

#===========================================#
#Hysteresis currve

ms = 1.23 * 10^6
Man(h) = 2 *  ms / pi * atan(norm(h)/38)
hs = [[600*sin(t),0] for t in range(0,2*π,1000)]
sol_prev = [0.0,0.0]
sols = zeros(length(hs) +1 ,2)
for (i,h) in enumerate(hs)
  p = Params2(71,h,sol_prev)
  up = ConstrainedProblem(p.χ, h, p.mₚ, test_fun_1_conjugate_builder(ms, 38))
  ui = Interface(up, p.mₚ + [0.1,0.0], 10, 1e-12)
  sol = solve(ui, :semi_smooth_newton, linesearch=BackTracking())
  sol_prev = sol.sol
  @info sol
  sols[i,:] .= Man(sol.sol) * sol.sol/norm(sol.sol)
end


f = Figure()
ax = Axis(f[1, 1])
lines!(ax,stack(hs)[1,:],sols[2:end,1])

f