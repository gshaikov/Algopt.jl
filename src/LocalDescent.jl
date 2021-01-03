module LocalDescent

using LinearAlgebra

import ..Bracketing:
BracketingSearch,
search_univariate

abstract type LocalDescentMethod end

# Line Search; Chapter 4.2

struct LineSearch <: LocalDescentMethod
    univariate_minimiser
    LineSearch(; univariate_minimiser=BracketingSearch()) = new(univariate_minimiser)
end

function search_local(params::LineSearch, f, x_k, d_k)
    """
    Line Search
    Algorithm 4.1
    """
    objective = α::Real -> f(x_k + α * d_k)::Real
    α_optimal = search_univariate(params.univariate_minimiser, objective, 0)
    x_k + α_optimal * d_k
end

# Strong Backtracking; Chapter 4.3

struct StrongBacktracking <: LocalDescentMethod
    β
    σ
    α0
    α_factor
    max_steps_find
    max_steps_zoom
    function StrongBacktracking(;
        β=1e-4, σ=0.1, α0=1, α_factor=2, max_steps_find=1_000_000, max_steps_zoom=10_000)
        new(β, σ, α0, α_factor, max_steps_find, max_steps_zoom)
    end
end

struct DescentPoint
    f_x  # f(x_k)
    ∇df_x  # ∇f(x_k) ⋅ d_k
    x  # x_k
    d  # d_k
    DescentPoint(f, ∇f, x_k, d_k) = new(f(x_k), ∇f(x_k) ⋅ d_k, x_k, d_k)
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
    function BracketConditions(params, f, ∇f, desc_point)
        new(α -> f(desc_point.x + α * desc_point.d) >= desc_point.f_x,
            α -> f(desc_point.x + α * desc_point.d) > desc_point.f_x + params.β * α * desc_point.∇df_x,
            α -> (∇f(desc_point.x + α * desc_point.d) ⋅ desc_point.d) >= 0)
    end
end

struct StrongWolfeConditions
    """
    Strong Wolfe conditions are termination conditions for local descent
        search of optimal optimization step α.
    """
    first
    second
    function StrongWolfeConditions(params, f, ∇f, desc_point)
        new(α -> f(desc_point.x + α * desc_point.d) <= desc_point.f_x + params.β * α * desc_point.∇df_x,
            α -> abs(∇f(desc_point.x + α * desc_point.d) ⋅ desc_point.d) <= -params.σ * desc_point.∇df_x)
    end
end

mutable struct MutBracket
    left
    right
end

function search_local(params::StrongBacktracking, f, ∇f, x_k, d_k)
    desc_point = DescentPoint(f, ∇f, x_k, d_k)
    bracket_cond = BracketConditions(params, f, ∇f, desc_point)
    wolfe_cond = StrongWolfeConditions(params, f, ∇f, desc_point)
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
        if bracket_cond.first(bracket.right) || bracket_cond.second(bracket.right) || bracket_cond.third(bracket.right)
            return bracket
        end
        bracket.left, bracket.right = bracket.right, params.α_factor * bracket.right
    end
    error("max number of steps reached: bracket $bracket, max_steps_find $(params.max_steps_find)")
end

function zoom_bracket!(bracket::MutBracket, params::StrongBacktracking, wolfe_cond::StrongWolfeConditions, is_ascending)::Real
    """
    Invariants
    - bracket contains a local minimum
    - gradient at bracket.left <= 0
    - bracket.left <= bracket.right
    """
    if wolfe_cond.first(bracket.left) && wolfe_cond.second(bracket.left)
        return bracket.left
    end
    if wolfe_cond.first(bracket.right) && wolfe_cond.second(bracket.right)
        return bracket.right
    end
    for _ = 1:params.max_steps_zoom
        if abs(bracket.left - bracket.right) <= eps()
            return bracket.right
        end
        α = bracket.left + (bracket.right - bracket.left) / 2
        if wolfe_cond.first(α) && wolfe_cond.second(α)
            return α
        end
        if is_ascending(α)
            bracket.right = α
        else
            bracket.left = α
        end
    end
    @warn "zoom_bracket: max number of steps reached: bracket $bracket, max_steps_zoom $(params.max_steps_zoom)"
    bracket.right
end

end # module
