module TestFunctionsTests

import ..Algopt # ensure we are using the correct Algopt

using Test

using Algopt.TestFunctions:
Rosenbrock

@testset "Algopt.TestFunctions.Rosenbrock" begin
    ros = Rosenbrock()
    @test 0 == ros.f(ros.argmin)
    @test [0, 0] == ros.∇f(ros.argmin)

    ros = Rosenbrock(a=100, b=16)
    @test 0 == ros.f(ros.argmin)
    @test [0, 0] == ros.∇f(ros.argmin)
end

end # module
