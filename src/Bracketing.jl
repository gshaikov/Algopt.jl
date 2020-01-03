module Bracketing

abstract type BracketingMethod end

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

function search(params::BracketingMethod, f, x_0 = 0)::Real
    bracket = find_bracket(f, x_0)
    search_bracket(params, f, bracket).right
end

function find_bracket(f, x; step = 1e-2, factor = 2.0, max_iter = 1_000_000)::Bracket
    """
    Algorithm 3.1

    `max_iter` is a very high number since upon reaching this limit the algorithm
    will crash with error. This is because a failure to find a bracket means there's
    something wrong with either the algorithm, or the objective function.
    """
    a, ya = x, f(x)
    b, yb = a + step, f(a + step)
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        step = step > 0 ? -step : step
    end
    for _ = 1:max_iter
        c, yc = b + step, f(b + step)
        if yc > yb
            return Bracket(a, c)
        end
        a, ya, b, yb = b, yb, c, yc
        step *= factor
    end
    error("Bracketing.find_bracket: max number of iterations reached")
end

struct GoldenSection <: BracketingMethod
    ϵ
    max_iter
    GoldenSection(; ϵ = eps(), max_iter = 100) = new(ϵ, max_iter)
end

function search_bracket(params::GoldenSection, f, bracket::Bracket)::Bracket
    """
    Golden section search
    Algorithm 3.3
    """
    a, b = bracket.left, bracket.right
    max_eval = (b - a) / (params.ϵ * log(Base.MathConstants.golden))
    n = params.max_iter > max_eval ? max_eval : params.max_iter

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

end # module
