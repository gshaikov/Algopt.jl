module FirstOrderTests

import ..Algopt # ensure we are using the correct Algopt

using Test

using LinearAlgebra

import Algopt.Bracketing:
BracketingSearch

import Algopt.LocalDescent:
StrongBacktracking

using Algopt.FirstOrder:
TerminationTolerance, MaximumGradientDescent, GradientDescent, ConjugateGradientDescent, CGDProblemState,
search, descent_step_maxgd, direction_maxgd, descent_step_gd, descent_step_cgd!

using Algopt.TestFunctions:
Rosenbrock


quad0 = x -> x'x
∇quad0 = x -> 2x

quad3 = x -> (x - [3, 3])' * (x - [3, 3])
∇quad3 = x -> 2(x - [3, 3])

ros = Rosenbrock(a=1, b=5)

@testset "maximum gradient descent" begin
    @test [-0.6, -0.8] == direction_maxgd([6, 8])
    @test [0, -1] == direction_maxgd([0, 2])

    @test [0, 0] == descent_step_maxgd(BracketingSearch(), quad0, ∇quad0, [3, 4])
    @test [0, 0] == descent_step_maxgd(BracketingSearch(), quad0, ∇quad0, [10, 10])
    @test [3, 3] ≈ descent_step_maxgd(BracketingSearch(), quad3, ∇quad3, [3, 4])
    @test [3, 3] ≈ descent_step_maxgd(BracketingSearch(), quad3, ∇quad3, [10, 10])

    mgd = MaximumGradientDescent()
    x0 = [rand(-10:.1:10), rand(-10:.1:10)]
    @test norm([0, 0] - search(mgd, quad0, ∇quad0, x0)) < cbrt(eps())
    @test norm([0, 0] - search(mgd, quad0, ∇quad0, x0)) < cbrt(eps())
    @test [3, 3] ≈ search(mgd, quad3, ∇quad3, x0)
    @test [3, 3] ≈ search(mgd, quad3, ∇quad3, x0)
    @test norm([1, 1] - search(mgd, ros.f, ros.∇f, x0)) < cbrt(eps())
end

@testset "gradient descent" begin
    α = 1e-2
    @test [3, 3.98] == descent_step_gd(α, ∇quad3, [3, 4])
    @test [9.86, 9.86] == descent_step_gd(α, ∇quad3, [10, 10])

    grd = GradientDescent(max_steps=100_000)
    x0 = [rand(-10:.1:10), rand(-10:.1:10)]
    @test norm([0, 0] - search(grd, quad0, ∇quad0, x0)) < cbrt(eps())
    @test norm([0, 0] - search(grd, quad0, ∇quad0, x0)) < cbrt(eps())
    @test norm([3, 3] - search(grd, quad3, ∇quad3, x0)) < cbrt(eps())
    @test norm([3, 3] - search(grd, quad3, ∇quad3, x0)) < cbrt(eps())
    @test norm([1, 1] - search(grd, ros.f, ros.∇f, x0)) < cbrt(eps())
end

@testset "conjugate gradient descent" begin
    @test [0, 0] == descent_step_cgd!(CGDProblemState(2), StrongBacktracking(), quad0, ∇quad0, [3, 4])
    @test [0, 0] == descent_step_cgd!(CGDProblemState(2), StrongBacktracking(), quad0, ∇quad0, [10, 10])
    @test [3, 3] == descent_step_cgd!(CGDProblemState(2), StrongBacktracking(), quad3, ∇quad3, [3, 4])
    @test [3, 3] == descent_step_cgd!(CGDProblemState(2), StrongBacktracking(), quad3, ∇quad3, [10, 10])

    cgd_quad = ConjugateGradientDescent(max_steps=100_000)
    x0_quad = [rand(-10:.1:10), rand(-10:.1:10)]
    @test [0, 0] ≈ search(cgd_quad, quad0, ∇quad0, x0_quad)
    @test [0, 0] ≈ search(cgd_quad, quad0, ∇quad0, x0_quad)
    @test [3, 3] ≈ search(cgd_quad, quad3, ∇quad3, x0_quad)
    @test [3, 3] ≈ search(cgd_quad, quad3, ∇quad3, x0_quad)

    cgd_ros = ConjugateGradientDescent(max_steps=100_000)
    x0_ros = [rand(-10:.1:10), rand(-10:.1:10)]
    @test norm([1, 1] - search(cgd_ros, ros.f, ros.∇f, x0_ros)) < cbrt(eps())
end

end # module
