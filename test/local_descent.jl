module LocalDescentTests

import ..Algopt # ensure we are using the correct Algopt

using Test

using Algopt.LocalDescent:
LineSearch, StrongBacktracking, DescentPoint, MutBracket, BracketConditions, StrongWolfeConditions,
search_local, find_bracket, zoom_bracket!

using Algopt.TestFunctions:
Rosenbrock


quad0 = x -> x'x
∇quad0 = x -> 2x

quad3 = x -> (x - [3, 3])' * (x - [3, 3])
∇quad3 = x -> 2(x - [3, 3])

@testset "line search with defaults" begin
    ls = LineSearch()
    @test [0, 0] ≈ search_local(ls, quad0, [10, 10], [-1, -1])
    @test [3, 3] ≈ search_local(ls, quad3, [10, 10], [-1, -1])

    ls = LineSearch()
    ros = Rosenbrock(a=1, b=5)
    @test [1, 1] ≈ search_local(ls, ros.f, [10, 10], [-1, -1])
    @test [1, 1] ≈ search_local(ls, ros.f, [-2, -2], [1, 1])
end

@testset "descent point" begin
    desc_point = DescentPoint(quad0, ∇quad0, [10,10], [-1,-1])
    @test desc_point.f_x == 200
    @test desc_point.∇df_x == -40  # [20, 20]' * [-1, -1]
    @test desc_point.x == [10, 10]
    @test desc_point.d == [-1, -1]
end

@testset "strong wolfe conditions" begin
    params = StrongBacktracking()
    desc_point = DescentPoint(quad0, ∇quad0, [10,10], [-1,-1])
    wolfe = StrongWolfeConditions(params, quad0, ∇quad0, desc_point)
    
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

@testset "bracket conditions" begin
    params = StrongBacktracking(; β=1e-1)
    desc_point = DescentPoint(quad0, ∇quad0, [10,10], [-1,-1])
    bracket = BracketConditions(params, quad0, ∇quad0, desc_point)

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

@testset "strong backtracking: find bracket" begin
    params = StrongBacktracking()
    desc_point = DescentPoint(quad0, ∇quad0, [10,0], [-1,0])
    bracket_cond = BracketConditions(params, quad0, ∇quad0, desc_point)
    bracket = find_bracket(params, bracket_cond)
    @test bracket.left < 10
    @test bracket.right > 10

    params = StrongBacktracking()
    desc_point = DescentPoint(quad0, ∇quad0, [0,-0.1], [0,1])
    bracket_cond = BracketConditions(params, quad0, ∇quad0, desc_point)
    bracket = find_bracket(params, bracket_cond)
    @test bracket.left < 0.1
    @test bracket.right > 0.1

    params = StrongBacktracking()
    desc_point = DescentPoint(quad0, ∇quad0, [-0.1,10], [0.01,-1])
    bracket_cond = BracketConditions(params, quad0, ∇quad0, desc_point)
    bracket = find_bracket(params, bracket_cond)
    @test bracket.left < 10
    @test bracket.right > 10

    params = StrongBacktracking()
    desc_point = DescentPoint(quad0, ∇quad0, [0.1,-10], [-0.01,1])
    bracket_cond = BracketConditions(params, quad0, ∇quad0, desc_point)
    bracket = find_bracket(params, bracket_cond)
    @test bracket.left < 10
    @test bracket.right > 10
end

@testset "strong backtracking: zoom bracket" begin
    params = StrongBacktracking()
    desc_point = DescentPoint(quad0, ∇quad0, [10,10], [-1,-1])
    bracket_cond = BracketConditions(params, quad0, ∇quad0, desc_point)
    wolfe_cond = StrongWolfeConditions(params, quad0, ∇quad0, desc_point)

    bracket = MutBracket(0, 20)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α == 10

    bracket = MutBracket(0, 16)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α == 10

    bracket = MutBracket(0, 12)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α == 9

    bracket = MutBracket(8, 20)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α == 11

    bracket = MutBracket(4, 12)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α == 10

    bracket = MutBracket(0, 11)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α == 11

    bracket = MutBracket(9, 20)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α == 9

    bracket = MutBracket(0, 10)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α == 10

    bracket = MutBracket(10, 20)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    @test α == 10
end

@testset "strong backtracking" begin
    sb = StrongBacktracking()
    @test [0, 0] ≈ search_local(sb, quad0, ∇quad0, [10, 10], [-1, -1])
    @test [0, 0] ≈ search_local(sb, quad0, ∇quad0, [-0.1, 10], [0.01, -1])
    @test [3, 3] ≈ search_local(sb, quad3, ∇quad3, [10, 0.1], [-1, 2.9 / 7])
    @test [3, 3] ≈ search_local(sb, quad3, ∇quad3, [10, -10], [-1, 13 / 7])

    sb = StrongBacktracking(σ=cbrt(eps()))
    ros = Rosenbrock(a=1, b=5)
    @test [1, 1] ≈ search_local(sb, ros.f, ros.∇f, [10, 10], [-1, -1])
    @test [1, 1] ≈ search_local(sb, ros.f, ros.∇f, [-2, -2], [1, 1])
end

end # module
