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

function descent_until(term::Termination, descent_method, f, ∇f, X, trace)
    term_cond = TerminationConditions(term)
    for _ = 1:term.max_iter
        x = X[:, end]
        x_next = descent_method(f, ∇f, x)
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

abstract type FirstOrderMethods end

abstract type ProblemState end

# Maximum Gradient Descent

struct MaximumGradientDescent <: FirstOrderMethods end

function search(::MaximumGradientDescent, f, ∇f, x_0; term = Termination(), trace = false)
    descent_until(term, descent_step_maxgd, f, ∇f, x_0, trace)
end

function descent_step_maxgd(f, ∇f, x)
    """
    Maximum Gradient Descent

    A step of maximum gradient descent produces the value of x(k+1)
    where f(x) in the direction of maximum descent is at its local
    minimum.
    """
    d = direction_maxgd(∇f, x)
    search(LineSearch(), f, x, d)
end

function direction_maxgd(∇f, x)
    g = ∇f(x)
    -g / norm(g)
end

# Gradient Descent

struct GradientDescent <: FirstOrderMethods
    α
    GradientDescent(; α = 0.001) = new(α)
end

function search(params::GradientDescent, f, ∇f, x_0; term = Termination(), trace = false)
    descent_step = (f, ∇f, x)->descent_step_gd(params.α, f, ∇f, x)
    descent_until(term, descent_step, f, ∇f, x_0, trace)
end

function descent_step_gd(α, f, ∇f, x)
    """
    Gradient Descent
    Algorithm 5.1

    A step of a gradient descent algorithm with fixed learning rate α.
    """
    g = ∇f(x)
    x - α * g
end

end # module
