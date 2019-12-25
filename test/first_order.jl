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

quad0 = x->x'x
∇quad0 = x->2x

quad3 = x->(x - [3, 3])' * (x - [3, 3])
∇quad3 = x->2(x - [3, 3])

@testset "Algopt.FirstOrder.MaximumGradientDescent" begin
    mgd = MaximumGradientDescent()

    @test [-0.6, -0.8] == direction_maxgd([6, 8])
    @test [0, -1] == direction_maxgd([0, 2])

    @test [0, 0] == descent_step_maxgd(quad0, ∇quad0, [3, 4])
    @test [0, 0] == descent_step_maxgd(quad0, ∇quad0, [10, 10])
    @test [3, 3] ≈ descent_step_maxgd(quad3, ∇quad3, [3, 4])
    @test [3, 3] ≈ descent_step_maxgd(quad3, ∇quad3, [10, 10])

    @test [0, 0] == search(mgd, quad0, ∇quad0, [3, 4])
    @test [0, 0] == search(mgd, quad0, ∇quad0, [10, 10])
    @test [3, 3] ≈ search(mgd, quad3, ∇quad3, [3, 4])
    @test [3, 3] ≈ search(mgd, quad3, ∇quad3, [10, 10])
end

@testset "Algopt.FirstOrder.GradientDescent" begin
    α = 1e-2
    gd = GradientDescent(α = α)

    @test [3, 3.98] ≈ descent_step_gd(α, quad3, ∇quad3, [3, 4])
    @test [9.86, 9.86] ≈ descent_step_gd(α, quad3, ∇quad3, [10, 10])

    @test norm([3, 3] - search(gd, quad3, ∇quad3, [3, 4])) < 1
    @test norm([3, 3] - search(gd, quad3, ∇quad3, [10, 10])) < 1
end

@testset "Algopt.FirstOrder.ConjugateGradientDescent" begin
    cgd = ConjugateGradientDescent()

    @test norm([0, 0] - descent_step_cgd!(CGDProblemState(2), quad0, ∇quad0, [3, 4])) < 1e-9
    @test norm([0, 0] - descent_step_cgd!(CGDProblemState(2), quad0, ∇quad0, [10, 10])) < 1e-9
    @test norm([3, 3] - descent_step_cgd!(CGDProblemState(2), quad3, ∇quad3, [3, 4])) < 1e-9
    @test norm([3, 3] - descent_step_cgd!(CGDProblemState(2), quad3, ∇quad3, [10, 10])) < 1e-9

    @test norm([0, 0] - search(cgd, quad0, ∇quad0, [3, 4])) < 1e-9
    @test norm([0, 0] - search(cgd, quad0, ∇quad0, [10, 10])) < 1e-9
    @test [3, 3] ≈ search(cgd, quad3, ∇quad3, [3, 4])
    @test [3, 3] ≈ search(cgd, quad3, ∇quad3, [10, 10])
end

end # module
