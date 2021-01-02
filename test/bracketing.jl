module BracketingTests

import ..Algopt # ensure we are using correct Algopt

using Test

using Algopt.Bracketing:
Bracket, BracketingSearch, FindBracket, GoldenSectionSearch,
search_univariate, find_bracket, narrow_bracket

using Algopt.TestFunctions:
Rosenbrock


quadratic_test_function = x -> (x - 2)^2

rosenbrock_test_function = Rosenbrock(a=1, b=5)

@testset "bracketing with golden section search" begin
    optimal_design_point = search_univariate(
        BracketingSearch(narrow_bracket=GoldenSectionSearch()),
        quadratic_test_function,
        0
    )
    @test 2 ≈ optimal_design_point

    optimal_design_point = search_univariate(
        BracketingSearch(narrow_bracket=GoldenSectionSearch()),
        quadratic_test_function,
        -1e6
    )
    @test 2 ≈ optimal_design_point

    optimal_design_point = search_univariate(
        BracketingSearch(narrow_bracket=GoldenSectionSearch(
            max_steps=2
        )),
        quadratic_test_function,
        -1e6
    )
    @test !(2 ≈ optimal_design_point)

    optimal_design_point = search_univariate(
        BracketingSearch(narrow_bracket=GoldenSectionSearch()),
        x -> x'x,
        0
    )
    @test abs(0 - optimal_design_point) <= eps()

    # Line search of α on 2D Rosenbrock function
    ros_line = α -> rosenbrock_test_function.f([10, 10] + α * [-1, -1])
    optimal_design_point = search_univariate(
        BracketingSearch(narrow_bracket=GoldenSectionSearch()),
        ros_line,
        0
    )
    @test 9 ≈ optimal_design_point
end

@testset "find bracket" begin
    bracket = find_bracket(FindBracket(), quadratic_test_function, 0)
    @test bracket.left < 2
    @test bracket.right > 2
    
    bracket = find_bracket(FindBracket(), quadratic_test_function, 10)
    @test bracket.left < 2
    @test bracket.right > 2
    
    bracket = find_bracket(FindBracket(), quadratic_test_function, -1e6)
    @test bracket.left < 2
    @test bracket.right > 2
end

@testset "narrow bracket with golden section search" begin
    bracket = narrow_bracket(GoldenSectionSearch(), quadratic_test_function, Bracket(0, 10))
    @test bracket.left ≈ 2
    @test bracket.right ≈ 2

    bracket = narrow_bracket(GoldenSectionSearch(), quadratic_test_function, Bracket(-1e6, 1e6))
    @test bracket.left ≈ 2
    @test bracket.right ≈ 2

    bracket = narrow_bracket(GoldenSectionSearch(max_steps=2), quadratic_test_function, Bracket(-1e6, 1e6))
    @test !(bracket.left ≈ 2)
    @test !(bracket.right ≈ 2)
end

end # module
