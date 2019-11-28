module TestFirstOrder

using Test

using Algopt.FirstOrder:
search, descent_step, direction,
MaximumGradientDescent

quad0 = x->x'x
∇quad0 = x->2x

quad3 = x->(x - [3, 3])' * (x - [3, 3])
∇quad3 = x->2(x - [3, 3])

@testset "MaximumGradientDescent" begin
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

end
