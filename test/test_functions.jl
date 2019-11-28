module TestTestFunctions

using Test

using Algopt.TestFunctions:
Rosenbrock

@testset "Rosenbrock" begin
    ros = Rosenbrock()
    @test 0 == ros.f(ros.argmin)
    @test [0, 0] == ros.∇f(ros.argmin)

    ros = Rosenbrock(a=100, b=16)
    @test 0 == ros.f(ros.argmin)
    @test [0, 0] == ros.∇f(ros.argmin)
end

end