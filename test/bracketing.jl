module BracketingTests

import ..Algopt # ensure we are using the correct Algopt

using Test
using Algopt.Bracketing:
search, find_bracket, search_bracket,
Bracket,
GoldenSection

using Algopt.TestFunctions: Rosenbrock


quad2 = x->(x - 2)^2

ros = Rosenbrock(a = 1, b = 5)

@testset "Algopt.Bracketing.search(::GoldenSection)" begin
    @test 2 ≈ search(GoldenSection(), quad2)
    @test 2 ≈ search(GoldenSection(), quad2, -1e6)
    @test !(2 ≈ search(GoldenSection(max_iter = 2), quad2))
    
    @test abs(0 - search(GoldenSection(), x->x'x)) <= eps()

    # Line search of α on 2D Rosenbrock function
    ros_line = α->ros.f([10, 10] + α * [-1, -1])
    @test 9 ≈ search(GoldenSection(), ros_line, 0)
end

@testset "Algopt.Bracketing.find_bracket" begin
    bracket = find_bracket(quad2, 0)
    @test bracket.left < 2
    @test bracket.right > 2
    
    bracket = find_bracket(quad2, 10)
    @test bracket.left < 2
    @test bracket.right > 2
    
    bracket = find_bracket(quad2, -1e6)
    @test bracket.left < 2
    @test bracket.right > 2
end

@testset "Algopt.Bracketing.search_bracket" begin
    bracket = search_bracket(GoldenSection(), quad2, Bracket(0, 10))
    @test bracket.left ≈ 2
    @test bracket.right ≈ 2

    bracket = search_bracket(GoldenSection(), quad2, Bracket(-1e6, 1e6))
    @test bracket.left ≈ 2
    @test bracket.right ≈ 2

    bracket = search_bracket(GoldenSection(max_iter = 2), quad2, Bracket(-1e6, 1e6))
    @test !(bracket.left ≈ 2)
    @test !(bracket.right ≈ 2)
end

end # module
