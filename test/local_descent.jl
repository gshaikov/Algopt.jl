module TestLocalDescent

using Test

using Algopt.LocalDescent:
search,
find_bracket, zoom_bracket!,
LineSearch, StrongBacktracking,
DescentPointK, MutBracket,
BracketConditions, StrongWolfeConditions


quad0 = x->x'x
∇quad0 = x->2x

quad3 = x->(x - [3, 3])' * (x - [3, 3])
∇quad3 = x->2(x - [3, 3])

@testset "search LineSearch" begin
    ls = LineSearch()
    @test [0, 0] ≈ search(ls, quad0, [10, 10], [-1, -1])
    @test [3, 3] ≈ search(ls, quad3, [10, 10], [-1, -1])
end

@testset "search StrongBacktracking" begin
    sb = StrongBacktracking()
    @test [0, 0] ≈ search(sb, quad0, ∇quad0, [10, 10], [-1, -1])
    @test [3, 3] ≈ search(sb, quad3, ∇quad3, [10, 10], [-1, -1])
end

@testset "zoom_bracket!" begin
    params = StrongBacktracking()
    point_k = DescentPointK(quad0, ∇quad0, [10,10], [-1,-1])
    wolfe_cond = StrongWolfeConditions(params, quad0, ∇quad0, point_k)
    bracket_cond = BracketConditions(wolfe_cond, quad0, ∇quad0, point_k)
    
    bracket = MutBracket(0, 20)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α ≈ 10
    
    bracket = MutBracket(0, 10)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α ≈ 10
    
    bracket = MutBracket(0, 12)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α > 9
    @test α < 11
    
    bracket = MutBracket(10, 20)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α > 10
    @test α < 11
    
    bracket = MutBracket(9, 20)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α > 10
    @test α < 11
end

@testset "find_bracket" begin
    params = StrongBacktracking()
    point_k = DescentPointK(quad0, ∇quad0, [10,10], [-1,-1])
    wolfe_cond = StrongWolfeConditions(params, quad0, ∇quad0, point_k)
    bracket_cond = BracketConditions(wolfe_cond, quad0, ∇quad0, point_k)
    bracket = find_bracket(params, bracket_cond)
    @test bracket.left < 10
    @test bracket.right > 10
end

@testset "DescentPointK" begin
    point_k = DescentPointK(quad0, ∇quad0, [10,10], [-1,-1])
    @test point_k.f == 200
    @test point_k.∇df == -40  # [20, 20]' * [-1, -1]
    @test point_k.x == [10, 10]
    @test point_k.d == [-1, -1]
end

@testset "StrongWolfeConditions" begin
    params = StrongBacktracking()
    point_k = DescentPointK(quad0, ∇quad0, [10,10], [-1,-1])
    wolfe = StrongWolfeConditions(params, quad0, ∇quad0, point_k)
    
    @test wolfe.first(0)
    @test wolfe.first(1)
    @test wolfe.first(10)

    @test !wolfe.second(1)
    @test !wolfe.second(8)
    @test wolfe.second(9)
    @test wolfe.second(10)
    @test wolfe.second(11)
    @test !wolfe.second(12)
    @test !wolfe.second(20)
end

@testset "BracketConditions" begin
    params = StrongBacktracking(; β = 1e-1)
    point_k = DescentPointK(quad0, ∇quad0, [10,10], [-1,-1])
    wolfe = StrongWolfeConditions(params, quad0, ∇quad0, point_k)
    bracket = BracketConditions(wolfe, quad0, ∇quad0, point_k)

    @test bracket.first(0)
    @test !bracket.first(1)
    @test !bracket.first(10)
    @test !bracket.first(19)
    @test bracket.first(20)
    @test bracket.first(21)

    @test !bracket.second(0)
    @test !bracket.second(10)
    @test !bracket.second(18)
    @test bracket.second(18.000001)
    @test bracket.second(20)

    @test !bracket.third(0)
    @test !bracket.third(9)
    @test bracket.third(10)
    @test bracket.third(11)
    @test bracket.third(20)
end

end