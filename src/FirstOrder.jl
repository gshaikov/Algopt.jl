module FirstOrder

using LinearAlgebra

import ..LocalDescent:
search,
LineSearch, StrongBacktracking

struct Termination
    max_iter
    ϵ_abs
    ϵ_rel
    ϵ_grad
    Termination(;
        max_iter = 10_000,
        ϵ_abs = eps(),
        ϵ_rel = eps(),
        ϵ_grad = eps()) =
        new(max_iter, ϵ_abs, ϵ_rel, ϵ_grad)
    Termination(ϵ;
        max_iter = 10_000) =
        new(max_iter, ϵ, ϵ, ϵ)
end

mutable struct TerminationConditions
    abs
    rel
    grad
    TerminationConditions(term::Termination) = 
    new((fx, fx_next)->fx - fx_next < term.ϵ_abs,
        (fx, fx_next)->fx - fx_next < term.ϵ_abs * abs(fx),
        ∇fx_next->norm(∇fx_next) < term.ϵ_grad)
end

function descent_until(term::Termination, descent_method, f, ∇f, X, trace)
    term_cond = TerminationConditions(term)
    for _ = 1:term.max_iter
        x = X[:, end]
        x_next = descent_method(f, ∇f, x)
        if trace
            X = hcat(X, x_next)
        else
            X = x_next
        end
        fx, fx_next, ∇fx_next = f(x), f(x_next), ∇f(x_next)
        if term_cond.abs(fx, fx_next) ||
            term_cond.rel(fx, fx_next) ||
            term_cond.grad(∇fx_next)
            return X
        end
    end
    @warn "descent_until: max number of iterations reached"
    X
end

abstract type FirstOrderMethods end

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
    g = ∇f(x)
    d = direction_maxgd(g)
    search(LineSearch(), f, x, d)
end

function direction_maxgd(g)
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

# Conjugate Gradient Descent

struct ConjugateGradientDescent <: FirstOrderMethods end

mutable struct CGDProblemState
    g
    d
    CGDProblemState(dim) = new(ones(dim), zeros(dim))
end

function search(params::ConjugateGradientDescent, f, ∇f, x_0; term = Termination(), trace = false)
    state = CGDProblemState(size(x_0))
    descent_step = (f, ∇f, x)->descent_step_cgd!(state, f, ∇f, x)
    descent_until(term, descent_step, f, ∇f, x_0, trace)
end

function descent_step_cgd!(state, f, ∇f, x)
    """
    Conjugate Gradient Descent
    Algorithm 5.2

    https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
    """
    g = ∇f(x)
    d = direction_cgd(g, state.g, state.d)
    state.g = g
    state.d = d
    search(StrongBacktracking(), f, ∇f, x, d)
end

function direction_cgd(g, gm1, dm1)
    """
    Polak-Ribiere update
    Notation:
        g is for g(k)
        dm1 is for d(k-1)
        gm1 indicates g(k-1)
    """
    β_PR = (g' * (g - gm1)) / (gm1'gm1)
    β = max(β_PR, 0)
    -g + β * dm1
end

end # module
