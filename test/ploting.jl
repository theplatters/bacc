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
  ax1 = Axis(fig[1, 1], xticks=1:maximum(size(data)))
  ax2 = Axis(fig[1, 2],  xticks=1:maximum(size(data)))
  for (l,d) in zip(["Semi-smooth Newton (unconstrained)", "Subgradient descent", "Proximal gradient"],eachrow(data[1:3,:]))
    lines!(ax1, d[d .!= -Inf], label=l)
  end
  for (l,d) in zip(["Newton", "Semi-smooth Newton (constrained)"],eachrow(data[4:5,:]))
    lines!(ax2,1:length(d[d .!= -Inf]), d[d .!= -Inf], label=l)
  end
  axislegend(ax1, "Methods", position=:lt)
  axislegend(ax2, "Methods", position=:lt)
  fig
end

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

  @info solve(ui, :semi_smooth_newton, linesearch=WolfePowell(), callback=calculate_error!(1))
  @info sol = solve(ui, :subgradientdescent, linesearch=WolfePowell(), callback=calculate_error!(2))
  @info sol = solve(ui, :proximal_gradient, linesearch=WolfePowell(), callback=calculate_error!(3))
  @info solve(ci, :newton, linesearch=WolfePowell(), callback=calculate_error!(4))
  @info solve(ci, :semi_smooth_newton, linesearch=Static(), callback=calculate_error!(5))
  data
end
mp = 1 * ones(3)
p = Params2(10, 10 * ones(3), mp)
data1 = test_problem(test_fun_1_builder(200, 60), test_fun_1_conjugate_builder(200, 60), p)
fig1 = plot_convergencerate(data1)
data2 = test_problem(test_fun_2_builder(200, 60), test_fun_2_conjugate_builder(200, 60), p)
fig2 = plot_convergencerate(data2)


mp = 1e6 * ones(3)
p = Params2(70, 600 * ones(3), zeros(3))
data3 = test_problem(test_fun_1_builder(1.23e6, 38), test_fun_1_conjugate_builder(1.23e6, 38), p)
fig3 = plot_convergencerate(data3)
data4 = test_problem(test_fun_2_builder(1.23e6, 38), test_fun_2_conjugate_builder(1.23e6, 38), p)
fig4 = plot_convergencerate(data4)

p = Params2(70, 10 * ones(3), zeros(3))
data3 = [Vector{Float64}(undef, 0) for i in 1:5]
begin
  A = Matrix(I,(3,3))
  b = 100 * ones(3)
  c = 0
  data3 = test_problem(quadratic_builder(A, b, c), quadratic_builder(A, b, c), p)
end

fig5 = plot_convergencerate(data3)

save("plots/cr2_lowh.png", fig2)
save("plots/cr1_highh.png", fig3)
save("plots/cr2_highh.png", fig4)
save("plots/cr_quadr.png", fig5)

#===========================================#
#Hysteresis currve

#Custom LineSearche

ms = 1.23 * 10^6
A = 38
Man(h) = 2 *  ms / pi * atan(norm(h)/A)
hs = [[180*sin(t),0.0] for t in range(0,2*π,200)]
sol_prev = [00.0,0.0]
sols = zeros(length(hs) ,2)
for (i,h) in enumerate(hs)
  p = Params2(71,h,sol_prev)
  problem = ConstrainedProblem(p.χ, h, p.mₚ,test_fun_2_builder(ms, A))
  interface = Interface(problem, h + rand(2), 30, 1e-10)
  sol = solve(interface, :semi_smooth_newton, linesearch=WolfePowell())
  sol_prev = Man(sol.sol) * sol.sol/norm(sol.sol)
  sols[i,:] .= sol_prev
end


f = Figure()

ax = Axis(f[1, 1])
lines!(ax,stack(hs)[1,:],sols[1:end,1])

f
save("plots/hysteresis_curve.png",f)