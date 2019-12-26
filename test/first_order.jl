module FirstOrderTests

import ..Algopt # ensure we are using the correct Algopt

using Test

using LinearAlgebra

using Algopt.FirstOrder:
search, descent_step, direction,
Termination,
MaximumGradientDescent, GradientDescent

quad0 = x->x'x
∇quad0 = x->2x

quad3 = x->(x - [3, 3])' * (x - [3, 3])
∇quad3 = x->2(x - [3, 3])

@testset "Algopt.FirstOrder.MaximumGradientDescent" begin
    mgd = MaximumGradientDescent()
    @test [-0.6, -0.8] == direction(mgd, ∇quad0, [3, 4])
    @test [0, 0] == descent_step(mgd, quad0, ∇quad0, [3, 4])
    @test [0, 0] == search(mgd, quad0, ∇quad0, [3, 4])

    @test [0, 0] == descent_step(mgd, quad0, ∇quad0, [10, 10])
    @test [0, 0] == search(mgd, quad0, ∇quad0, [10, 10])

    mgd = MaximumGradientDescent()
    @test [0, -1] == direction(mgd, ∇quad3, [3, 4])
    @test [3, 3] ≈ descent_step(mgd, quad3, ∇quad3, [3, 4])
    @test [3, 3] ≈ search(mgd, quad3, ∇quad3, [3, 4])

    @test [3, 3] ≈ descent_step(mgd, quad3, ∇quad3, [10, 10])
    @test [3, 3] ≈ search(mgd, quad3, ∇quad3, [10, 10])
end

@testset "Algopt.FirstOrder.GradientDescent" begin
    gd = GradientDescent(α = 1e-2)
    @test [3, 3.98] ≈ descent_step(gd, quad3, ∇quad3, [3, 4])
    @test norm([3, 3] - search(gd, quad3, ∇quad3, [3, 4])) < 1

    @test [9.86, 9.86] ≈ descent_step(gd, quad3, ∇quad3, [10, 10])
    @test norm([3, 3] - search(gd, quad3, ∇quad3, [10, 10])) < 1
end

end # module
