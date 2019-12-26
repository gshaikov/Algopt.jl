module FirstOrder

using LinearAlgebra

import ..LocalDescent:
search,
LineSearch

struct Termination
    max_iter
    ϵ_abs
    ϵ_rel
    ϵ_grad
    Termination(;
        max_iter = 1_000,
        ϵ_abs = 1e-3,
        ϵ_rel = 1e-3,
        ϵ_grad = 1e-3) =
        new(max_iter, ϵ_abs, ϵ_rel, ϵ_grad)
    Termination(ϵ;
        max_iter = 1_000) =
        new(max_iter, ϵ, ϵ, ϵ)
end

mutable struct TerminationConditions
    abs
    rel
    grad
    TerminationConditions(term::Termination) = 
    new((fx, fx_next)->fx - fx_next < term.ϵ_abs,
        (fx, fx_next)->fx - fx_next < term.ϵ_abs * fx,
        ∇fx_next->norm(∇fx_next) < term.ϵ_grad)
end

abstract type FirstOrderMethods end

function search(params::FirstOrderMethods, f, ∇f, x_0; term = Termination(), trace = false)
    term_cond = TerminationConditions(term)
    X = x_0
    for _ = 1:term.max_iter
        x = X[:, end]
        x_next = descent_step(params, f, ∇f, x)
        fx, fx_next, ∇fx_next = f(x), f(x_next), ∇f(x_next)
        if term_cond.abs(fx, fx_next) ||
            term_cond.rel(fx, fx_next) ||
            term_cond.grad(∇fx_next)
            if trace
                X = hcat(X, x_next)
            else
                X = x_next
            end
            break
        end
        if trace
            X = hcat(X, x_next)
        else
            X = x_next
        end
    end
    X
end

struct MaximumGradientDescent <: FirstOrderMethods end

function descent_step(params::MaximumGradientDescent, f, ∇f, x)
    """
    Maximum Gradient Descent

    A step of maximum gradient descent produces the value of x(k+1)
    where f(x) in the direction of maximum descent is at its local
    minimum.
    """
    d = direction(params, ∇f, x)
    search(LineSearch(), f, x, d)
end

function direction(params::MaximumGradientDescent, ∇f, x)
    g = ∇f(x)
    -g / norm(g)
end

struct GradientDescent <: FirstOrderMethods
    α
    GradientDescent(; α = 1) = new(α)
end

function descent_step(params::GradientDescent, f, ∇f, x)
    """
    Gradient Descent
    Algorithm 5.1

    A step of a gradient descent algorithm with fixed learning rate α.
    """
    g = ∇f(x)
    x - params.α * g
end

end # module
