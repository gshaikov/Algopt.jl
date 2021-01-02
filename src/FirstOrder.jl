module FirstOrder

using LinearAlgebra

import ..Bracketing:
BracketingSearch

import ..LocalDescent:
StrongBacktracking,
search_local

struct TerminationTolerance
    ϵ_abs
    ϵ_rel
    ϵ_grad
    TerminationTolerance(;
        ϵ_abs=eps(),
        ϵ_rel=eps(),
        ϵ_grad=eps()) =
        new(ϵ_abs, ϵ_rel, ϵ_grad)
    TerminationTolerance(ϵ) = new(ϵ, ϵ, ϵ)
end

mutable struct TerminationPredicates
    abs
    rel
    grad
    TerminationPredicates(term::TerminationTolerance) = 
    new((fx, fx_next) -> fx - fx_next < term.ϵ_abs,
        (fx, fx_next) -> fx - fx_next < term.ϵ_abs * abs(fx),
        ∇fx_next -> norm(∇fx_next) < term.ϵ_grad)
end

function create_termination_checker(term, f, ∇f)
    term_pred = TerminationPredicates(term)
    function inner(x, x_next)
        fx, fx_next, ∇fx_next = f(x), f(x_next), ∇f(x_next)
        term_pred.abs(fx, fx_next) || term_pred.rel(fx, fx_next) || term_pred.grad(∇fx_next)
    end
    inner
end

function descent_until(descent_step, x, termination_tolerance_ok, max_steps, trace)
    x_trace = x
    for _ = 1:max_steps
        x_next = descent_step(x)
        if trace
            x_trace = hcat(x_trace, x_next)
        end
        if termination_tolerance_ok(x, x_next)
            return trace ? x_trace : x_next
        end
        x = x_next
    end
    @warn "descent_until: max number of iterations reached"
    trace ? x_trace : x
end

abstract type FirstOrderMethod end

# Maximum Gradient Descent

struct MaximumGradientDescent <: FirstOrderMethod
    local_search
    term
    max_steps
    MaximumGradientDescent(;
        local_search=BracketingSearch(),
        term=TerminationTolerance(),
        max_steps=10_000) =
        new(local_search, term, max_steps)
end

function search(params::MaximumGradientDescent, f, ∇f, x_0; trace=false)
    descent_step = (x) -> descent_step_maxgd(params.local_search, f, ∇f, x)
    descent_until(
        descent_step,
        x_0,
        create_termination_checker(params.term, f, ∇f),
        params.max_steps,
        trace
    )
end

function descent_step_maxgd(params, f, ∇f, x)
    """
    Maximum Gradient Descent

    A step of maximum gradient descent produces the value of x(k+1)
    where f(x) in the direction of maximum descent is at its local
    minimum.
    """
    g = ∇f(x)
    d = direction_maxgd(g)
    search_local(params, f, x, d)
end

function direction_maxgd(g)
    -g / norm(g)
end

# Gradient Descent

struct GradientDescent <: FirstOrderMethod
    α
    term
    max_steps
    GradientDescent(;
        α=0.001,
        term=TerminationTolerance(),
        max_steps=10_000) =
        new(α, term, max_steps)
end

function search(params::GradientDescent, f, ∇f, x_0; trace=false)
    descent_step = (x) -> descent_step_gd(params.α, ∇f, x)
    descent_until(
        descent_step,
        x_0,
        create_termination_checker(params.term, f, ∇f),
        params.max_steps,
        trace
    )
end

function descent_step_gd(α, ∇f, x)
    """
    Gradient Descent
    Algorithm 5.1

    A step of a gradient descent algorithm with fixed learning rate α.
    """
    g = ∇f(x)
    x - α * g
end

# Conjugate Gradient Descent

struct ConjugateGradientDescent <: FirstOrderMethod
    local_search
    term
    max_steps
    ConjugateGradientDescent(;
        local_search=StrongBacktracking(),
        term=TerminationTolerance(),
        max_steps=10_000) =
        new(local_search, term, max_steps)
end

mutable struct CGDProblemState
    g
    d
    CGDProblemState(dim) = new(ones(dim), zeros(dim))
end

function search(params::ConjugateGradientDescent, f, ∇f, x_0; trace=false)
    state = CGDProblemState(size(x_0))
    descent_step = (x) -> descent_step_cgd!(state, params.local_search, f, ∇f, x)
    descent_until(
        descent_step,
        x_0,
        create_termination_checker(params.term, f, ∇f),
        params.max_steps,
        trace
    )
end

function descent_step_cgd!(state, params, f, ∇f, x)
    """
    Conjugate Gradient Descent
    Algorithm 5.2

    https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
    """
    g = ∇f(x)
    d = direction_cgd(g, state.g, state.d)
    state.g = g
    state.d = d
    search_local(params, f, ∇f, x, d)
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
