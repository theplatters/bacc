using Test
using Bachelorarbeit
using LinearAlgebra

## Assert, that functions are really convex conjugates
include("testfunctions.jl")




struct Params2
    χ::Float64
    h::Vector{Float64}
    mₚ::Vector{Float64}
end

function test_problem(fun, fun_conv, p::Params2)
	up = UnconstrainedProblem(p.χ, p.h, p.mₚ, fun)
	ui = Interface(up, p.mₚ, 100, 1e-8)
	cp = ConstrainedProblem(p.χ, p.h, p.mₚ, fun_conv)
	ci = Interface(cp, p.h, 100, 1e-8)


	uc = solve(ci, :semi_smooth_newton, linesearch = BackTracking()).sol 
    uuc = solve(ui, :semi_smooth_newton, linesearch = BackTracking()).sol
    norm(cp.∇S(uc) - uuc) ≤ 1e-9 && norm(up.∇U(uuc) - uc) ≤ 1e-9
end
@testset "convex_conjugates" begin
    us = rand(3, 100)
    a = 60
    m = 80
    χ = 16
    h = 10 * ones(3)
    mp = [0.1,0.2,0.01] 
    @testset "Fun1" begin 
        cp = ConstrainedProblem(χ, h, mp, test_fun_1_builder(m, a))
        up = UnconstrainedProblem(χ, h, mp, test_fun_1_conjugate_builder(m, a))
        @test all(norm.([cp.∇S(up.∇U(u)) - u for u in eachcol(us)]) .≤ 1e-9)
    end
    @testset "Fun2" begin 
        cp = ConstrainedProblem(χ, h, mp, test_fun_2_builder(m, a))
        up = UnconstrainedProblem(χ, h, mp, test_fun_2_conjugate_builder(m, a))
        @test all(norm.([cp.∇S(up.∇U(u)) - u for u in eachcol(us)]) .≤ 1e-9)
    end   
    @testset "Quadr" begin 
        A = randn(3,3); A = A'*A; A = (A + A')/2
        b = rand(3)
        c = 5
        cp = ConstrainedProblem(χ, h, mp, quadratic_builder(A,b,c))
        up = UnconstrainedProblem(χ, h, mp, quadratic_conjugate_builder(A,b,c))
        @test all(norm.([cp.∇S(up.∇U(u)) - u for u in eachcol(us)]) .≤ 1e-9)
    end
end

@testset "convex_conjugate_solutions" begin
   @test test_problem(test_fun_1_builder(80, 60), test_fun_1_conjugate_builder(80, 60), Params2(16, 10 * ones(3), [0.1, 0.01, 0.2])) 
   @test test_problem(test_fun_2_builder(80, 60), test_fun_2_conjugate_builder(80, 60), Params2(16, 10 * ones(3), [0.1, 0.01, 0.2])) 

   @test test_problem(test_fun_1_builder(80, 60), test_fun_1_conjugate_builder(80, 60), Params2(16, 1 * ones(3), [0.1, 0.01, 0.2])) 
   @test test_problem(test_fun_2_builder(80, 60), test_fun_2_conjugate_builder(80, 60), Params2(16, 1 * ones(3), [0.1, 0.01, 0.2])) 
   
end
