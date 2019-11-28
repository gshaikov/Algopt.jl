module LocalDescent

include("./bracketing.jl")

using LinearAlgebra
import .Bracketing:
search,
GoldenSection

abstract type LocalDescentMethods end

struct LineSearch <: LocalDescentMethods
    max_iter
    LineSearch(; max_iter = 1_000) = new(max_iter)
end

function search(params::LineSearch, f, x_k, d_k)
    """
    Line Search
    Algorithm 4.1
    """
    objective = α->f(x_k + α * d_k)::Real
    α_star = search(GoldenSection(max_iter = params.max_iter), objective, 0)
    x_k + α_star * d_k
end

struct StrongBacktracking <: LocalDescentMethods
    α0
    β
    σ
    α_factor
    max_iter_bracketing
    max_iter_zooming
    function StrongBacktracking(;
        α0 = 1, β = 1e-4, σ = 0.1, α_factor = 2,
        max_iter_bracketing = 1_000, max_iter_zooming = 1_000)
        new(α0, β, σ, α_factor, max_iter_bracketing, max_iter_zooming)
    end
end

struct DescentPointK
    f  # f(x_k)
    ∇df  # ∇f(x_k) ⋅ d_k
    x  # x_k
    d  # d_k
    DescentPointK(f, ∇f, x_k, d_k) = new(f(x_k), ∇f(x_k) ⋅ d_k, x_k, d_k)
end

struct StrongWolfeConditions
    """
    Strong Wolfe conditions are termination conditions for local descent
        search of optimal optimization step α.
    """
    first
    second
    function StrongWolfeConditions(params, f, ∇f, point_k)
        new(α->f(point_k.x + α * point_k.d) <= point_k.f + params.β * α * point_k.∇df,
            α->abs(∇f(point_k.x + α * point_k.d) ⋅ point_k.d) <= -params.σ * point_k.∇df)
    end
end

struct BracketConditions
    """
    Bracket conditions are termination condition for a search of a bracket
        [α_left, α_right] such that the bracket contains a local minimum of
        the function f(x + αd).
    """
    first
    second
    third
    function BracketConditions(wolfe_cond, f, ∇f, point_k)
        new(α->f(point_k.x + α * point_k.d) >= point_k.f,
            α->!wolfe_cond.first(α),
            α->(∇f(point_k.x + α * point_k.d) ⋅ point_k.d) >= 0)
    end
end

mutable struct MutBracket
    left
    right
end

function search(params::StrongBacktracking, f, ∇f, x_k, d_k)
    point_k = DescentPointK(f, ∇f, x_k, d_k)
    wolfe_cond = StrongWolfeConditions(params, f, ∇f, point_k)
    bracket_cond = BracketConditions(wolfe_cond, f, ∇f, point_k)
    bracket = find_bracket(params, bracket_cond)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    x_k + α * d_k
end

function find_bracket(params, bracket_cond)::MutBracket
    bracket = MutBracket(0, params.α0)
    for iters = 1:params.max_iter_bracketing
        if bracket_cond.first(bracket.right) ||
            bracket_cond.second(bracket.right) ||
            bracket_cond.third(bracket.right)
            break
        end
        if iters == params.max_iter_bracketing
            error("max number of bracketing iterations reached")
        end
        bracket.left, bracket.right = bracket.right, params.α_factor * bracket.right
    end
    bracket
end

function zoom_bracket!(bracket::MutBracket, params, wolfe_cond, is_ascending)::Real
    for iters = 1:params.max_iter_zooming
        if wolfe_cond.first(bracket.right) &&
            wolfe_cond.second(bracket.right)
            break
        end
        α = bracket.left + (bracket.right - bracket.left) / 2
        if is_ascending(α)
            bracket.right = α
        else
            bracket.left = α
        end
    end
    bracket.right
end

end
