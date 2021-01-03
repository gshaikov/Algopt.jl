module BracketingTests

import ..Algopt # ensure we are using correct Algopt

using Test

using Algopt.Bracketing:
Bracket, BracketingSearch, FindBracket, GoldenSectionSearch,
search_univariate, find_bracket, narrow_bracket

using Algopt.TestFunctions:
Rosenbrock

@testset "bracketing with golden section search" begin
    optimal_design_point = search_univariate(
        BracketingSearch(narrow_bracket=GoldenSectionSearch()),
        x -> (x - 2)^2,
        0
    )
    @test 2 ≈ optimal_design_point

    optimal_design_point = search_univariate(
        BracketingSearch(narrow_bracket=GoldenSectionSearch()),
        x -> (x - 1e3)^2,
        -1e6
    )
    @test 1e3 ≈ optimal_design_point

    optimal_design_point = search_univariate(
        BracketingSearch(narrow_bracket=GoldenSectionSearch()),
        x -> x'x,
        0
    )
    @test isapprox(0, optimal_design_point, atol=eps())

    # Line search of α on 2D Rosenbrock function
    ros_line = α -> Rosenbrock(a=1, b=5).f([10, 10] + α * [-1, -1])
    optimal_design_point = search_univariate(
        BracketingSearch(narrow_bracket=GoldenSectionSearch()),
        ros_line,
        0
    )
    @test 9 ≈ optimal_design_point
end

@testset "find bracket" begin
    quad2 = x -> (x - 2)^2

    bracket = find_bracket(FindBracket(), quad2, 0)
    @test bracket.left < 2
    @test bracket.right > 2
    
    bracket = find_bracket(FindBracket(), quad2, 10)
    @test bracket.left < 2
    @test bracket.right > 2
    
    bracket = find_bracket(FindBracket(), quad2, -1e6)
    @test bracket.left < 2
    @test bracket.right > 2
end

@testset "narrow bracket with golden section search" begin
    bracket = narrow_bracket(GoldenSectionSearch(), x -> (x - 2)^2, Bracket(0, 10))
    @test bracket.left ≈ 2
    @test bracket.right ≈ 2

    bracket = narrow_bracket(GoldenSectionSearch(), x -> (x - 3.14)^2, Bracket(-1e6, 1e6))
    @test bracket.left ≈ 3.14
    @test bracket.right ≈ 3.14
end

end # module
