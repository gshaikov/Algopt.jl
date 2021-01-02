module LocalDescent

using LinearAlgebra

import ..Bracketing:
BracketingSearch,
search_univariate

abstract type LocalDescentMethod end

# Line Search; Chapter 4.2

function search_local(params::BracketingSearch, f, x_k, d_k)
    """
    Line Search
    Algorithm 4.1
    """
    objective = α::Real -> f(x_k + α * d_k)::Real
    α_optimal = search_univariate(params, objective, 0)
    x_k + α_optimal * d_k
end

# Strong Backtracking; Chapter 4.3

struct StrongBacktracking <: LocalDescentMethod
    α0
    β
    σ
    α_factor
    max_steps_find
    max_steps_zoom
    function StrongBacktracking(; α0=1, β=1e-4, σ=0.1, α_factor=2, max_steps_find=1_000_000, max_steps_zoom=1_000)
        new(α0, β, σ, α_factor, max_steps_find, max_steps_zoom)
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
        new(α -> f(point_k.x + α * point_k.d) <= point_k.f + params.β * α * point_k.∇df,
            α -> abs(∇f(point_k.x + α * point_k.d) ⋅ point_k.d) <= -params.σ * point_k.∇df)
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
        new(α -> f(point_k.x + α * point_k.d) >= point_k.f,
            α -> !wolfe_cond.first(α),
            α -> (∇f(point_k.x + α * point_k.d) ⋅ point_k.d) >= 0)
    end
end

mutable struct MutBracket
    left
    right
end

function search_local(params::StrongBacktracking, f, ∇f, x_k, d_k)
    point_k = DescentPointK(f, ∇f, x_k, d_k)
    wolfe_cond = StrongWolfeConditions(params, f, ∇f, point_k)
    bracket_cond = BracketConditions(wolfe_cond, f, ∇f, point_k)
    bracket = find_bracket(params, bracket_cond)
    α = zoom_bracket!(bracket, params, wolfe_cond, bracket_cond.third)
    x_k + α * d_k
end

function find_bracket(params::StrongBacktracking, bracket_cond)::MutBracket
    """
    `max_steps` is a very high number since upon reaching this limit the algorithm
    will crash with error. This is because a failure to find a bracket means there's
    something wrong with either the algorithm, or the objective function.
    """
    bracket = MutBracket(0, params.α0)
    for _ = 1:params.max_steps_find
        if bracket_cond.first(bracket.right) ||
            bracket_cond.second(bracket.right) ||
            bracket_cond.third(bracket.right)
            return bracket
        end
        bracket.left, bracket.right = bracket.right, params.α_factor * bracket.right
    end
    error("max number of iterations reached")
end

function zoom_bracket!(bracket::MutBracket, params::StrongBacktracking, wolfe_cond::StrongWolfeConditions, is_ascending)::Real
    for _ = 1:params.max_steps_zoom
        if (wolfe_cond.first(bracket.right) &&
            wolfe_cond.second(bracket.right)) ||
            abs(bracket.left - bracket.right) < eps()
            return bracket.right
        end
        α = bracket.left + (bracket.right - bracket.left) / 2
        if is_ascending(α)
            bracket.right = α
        else
            bracket.left = α
        end
    end
    @warn "zoom_bracket: max number of iterations reached: bracket: $bracket"
    bracket.right
end

end # module
