module Bracketing

struct Bracket
    left::Real
    right::Real

    function Bracket(a, b)
        if a <= b
            new(a, b)
        else
            new(b, a)
        end
    end
end

struct FindBracket
    initial_step_size
    step_growth_factor
    max_steps
    FindBracket(; initial_step_size=1e-2, step_growth_factor=2.0, max_steps=1_000_000) = new(initial_step_size, step_growth_factor, max_steps)
end

abstract type NarrowBracketMethod end

struct GoldenSectionSearch <: NarrowBracketMethod
    ϵ
    max_steps
    GoldenSectionSearch(; ϵ=eps(), max_steps=10_000) = new(ϵ, max_steps)
end

struct BracketingSearch
    find_bracket::FindBracket
    narrow_bracket::NarrowBracketMethod
    BracketingSearch(; find_bracket=FindBracket(), narrow_bracket=GoldenSectionSearch()) = new(find_bracket, narrow_bracket)
end

function search_univariate(params::BracketingSearch, f, x_0)::Real
    """
    Find an approximate local minimum of a univariate function f.
    """
    bracket = find_bracket(params.find_bracket, f, x_0)
    bracket = narrow_bracket(params.narrow_bracket, f, bracket)
    bracket.right
end

function find_bracket(params::FindBracket, f, x)::Bracket
    """
    Find a bracket that contains a minimum of f.
    Algorithm 3.1

    `max_steps` is a very high number since upon reaching this limit the algorithm
    will crash with error. This is because a failure to find a bracket means there's
    something wrong with either the algorithm, or the objective function.
    """
    step_size = params.initial_step_size
    a, ya = x, f(x)
    b, yb = a + step_size, f(a + step_size)
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        step_size = step_size > 0 ? -step_size : step_size
    end
    for _ = 1:params.max_steps
        c, yc = b + step_size, f(b + step_size)
        if yc > yb
            return Bracket(a, c)
        end
        a, ya, b, yb = b, yb, c, yc
        step_size *= params.step_growth_factor
    end
    error("maximum number of steps reached")
end

function narrow_bracket(params::GoldenSectionSearch, f, bracket::Bracket)::Bracket
    """
    Golden section search
    Algorithm 3.3

    Narrow the bracket around the minimum.
    """
    a, b = bracket.left, bracket.right
    max_eval = compute_max_eval(a, b, params.ϵ)
    if params.max_steps >= max_eval
        n = max_eval
    else
        @warn "narrow_bracket: bracket too wide: bracket $(bracket), max_eval $max_eval, max_steps $(params.max_steps)"
        n = params.max_steps
    end

    ρ = Base.MathConstants.golden - 1
    d = ρ * b + (1 - ρ) * a
    yd = f(d)
    for _ = 1:n - 1
        if abs(a - b) < params.ϵ
            return Bracket(a, b)
        end
        c = ρ * a + (1 - ρ) * b
        yc = f(c)
        if yc < yd
            b, d, yd = d, c, yc
        else
            a, b = b, c
        end
    end
    Bracket(a, b)
end

function compute_max_eval(a, b, ϵ)
    bae = (b - a) / ϵ
    1 + log(bae) / log(Base.MathConstants.golden)
end

end # module
