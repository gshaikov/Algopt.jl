module TestBracketing

using Test
using Algopt.Bracketing:
search, find_bracket, search_bracket,
Bracket,
GoldenSection

quadratic = x->(x - 2)^2

@testset "search" begin
    @test 2 ≈ search(GoldenSection(), quadratic)
    @test 2 ≈ search(GoldenSection(), quadratic, -1e6)
    @test !(2 ≈ search(GoldenSection(max_iter = 2), quadratic))
    
    @test abs(0 - search(GoldenSection(), x->x'x)) < eps()

    quad = x->x'x
    direct = α->quad([10, 10] + α * [-1, -1])
    @test 10 ≈ search(GoldenSection(), direct, 0)
end

@testset "find_bracket" begin
    bracket = find_bracket(quadratic, 0)
    @test bracket.left < 2
    @test bracket.right > 2
    
    bracket = find_bracket(quadratic, 10)
    @test bracket.left < 2
    @test bracket.right > 2
    
    bracket = find_bracket(quadratic, -1e6)
    @test bracket.left < 2
    @test bracket.right > 2
end

@testset "search_bracket" begin
    bracket = search_bracket(quadratic, Bracket(0, 10), GoldenSection())
    @test bracket.left ≈ 2
    @test bracket.right ≈ 2

    bracket = search_bracket(quadratic, Bracket(-1e6, 1e6), GoldenSection())
    @test bracket.left ≈ 2
    @test bracket.right ≈ 2

    bracket = search_bracket(quadratic, Bracket(-1e6, 1e6), GoldenSection(max_iter = 2))
    @test !(bracket.left ≈ 2)
    @test !(bracket.right ≈ 2)
end

end