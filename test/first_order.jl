module FirstOrderTests

import ..Algopt # ensure we are using the correct Algopt

using Test

using LinearAlgebra

using Algopt.FirstOrder:
Termination,
MaximumGradientDescent, GradientDescent,
ConjugateGradientDescent, CGDProblemState,
search,
descent_step_maxgd, direction_maxgd,
descent_step_gd,
descent_step_cgd!

using Algopt.TestFunctions: Rosenbrock


quad0 = x->x'x
∇quad0 = x->2x

quad3 = x->(x - [3, 3])' * (x - [3, 3])
∇quad3 = x->2(x - [3, 3])

ros = Rosenbrock(a = 1, b = 5)

@testset "Algopt.FirstOrder.MaximumGradientDescent" begin
    @test [-0.6, -0.8] == direction_maxgd([6, 8])
    @test [0, -1] == direction_maxgd([0, 2])

    @test [0, 0] == descent_step_maxgd(quad0, ∇quad0, [3, 4])
    @test [0, 0] == descent_step_maxgd(quad0, ∇quad0, [10, 10])
    @test [3, 3] ≈ descent_step_maxgd(quad3, ∇quad3, [3, 4])
    @test [3, 3] ≈ descent_step_maxgd(quad3, ∇quad3, [10, 10])

    mgd = MaximumGradientDescent()
    x0 = [rand(-10:.1:10), rand(-10:.1:10)]
    @test norm([0, 0] - search(mgd, quad0, ∇quad0, x0)) < cbrt(eps())
    @test norm([0, 0] - search(mgd, quad0, ∇quad0, x0)) < cbrt(eps())
    @test [3, 3] ≈ search(mgd, quad3, ∇quad3, x0)
    @test [3, 3] ≈ search(mgd, quad3, ∇quad3, x0)
    @test norm([1, 1] - search(mgd, ros.f, ros.∇f, x0)) < cbrt(eps())
end

@testset "Algopt.FirstOrder.GradientDescent" begin
    α = 1e-2
    @test [3, 3.98] == descent_step_gd(α, quad3, ∇quad3, [3, 4])
    @test [9.86, 9.86] == descent_step_gd(α, quad3, ∇quad3, [10, 10])

    grd = GradientDescent()
    x0 = [rand(-10:.1:10), rand(-10:.1:10)]
    term = Termination(max_iter = 100_000)
    @test norm([0, 0] - search(grd, quad0, ∇quad0, x0; term = term)) < cbrt(eps())
    @test norm([0, 0] - search(grd, quad0, ∇quad0, x0; term = term)) < cbrt(eps())
    @test norm([3, 3] - search(grd, quad3, ∇quad3, x0; term = term)) < cbrt(eps())
    @test norm([3, 3] - search(grd, quad3, ∇quad3, x0; term = term)) < cbrt(eps())
    @test norm([1, 1] - search(grd, ros.f, ros.∇f, x0; term = term)) < cbrt(eps())
end

@testset "Algopt.FirstOrder.ConjugateGradientDescent" begin
    @test [0, 0] == descent_step_cgd!(CGDProblemState(2), quad0, ∇quad0, [3, 4])
    @test [0, 0] == descent_step_cgd!(CGDProblemState(2), quad0, ∇quad0, [10, 10])
    @test [3, 3] == descent_step_cgd!(CGDProblemState(2), quad3, ∇quad3, [3, 4])
    @test [3, 3] == descent_step_cgd!(CGDProblemState(2), quad3, ∇quad3, [10, 10])

    cgd = ConjugateGradientDescent()
    x0 = [rand(-10:.1:10), rand(-10:.1:10)]
    @test [0, 0] ≈ search(cgd, quad0, ∇quad0, x0)
    @test [0, 0] ≈ search(cgd, quad0, ∇quad0, x0)
    @test [3, 3] ≈ search(cgd, quad3, ∇quad3, x0)
    @test [3, 3] ≈ search(cgd, quad3, ∇quad3, x0)
    @test norm([1, 1] - search(cgd, ros.f, ros.∇f, x0)) < cbrt(eps())
end

end # module
