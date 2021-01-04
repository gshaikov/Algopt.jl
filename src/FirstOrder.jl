module FirstOrder

using LinearAlgebra

import ..LocalDescent:
LocalDescentMethod, LineSearch, StrongBacktracking,
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

function check_termination(term::TerminationTolerance, f, ∇f, x, x_next)
    fx, fx_next, ∇fx_next = f(x), f(x_next), ∇f(x_next)
    (
        fx - fx_next < term.ϵ_abs ||
        fx - fx_next < term.ϵ_rel * abs(fx) ||
        norm(∇fx_next) < term.ϵ_grad
    )
    # && fx - fx_next >= 0
end

mutable struct TrainingLog
    x
    fx
    ∇fx
end

function log_step!(tl::TrainingLog, f, ∇f, x)
    fx, ∇fx = f(x), ∇f(x)
    tl.x = hcat(tl.x, x)
    tl.fx = hcat(tl.fx, fx)
    tl.∇fx = hcat(tl.∇fx, ∇fx)
end

function descent_until(descent_step, x, should_terminate, max_steps, log_step)
    for _ = 1:max_steps
        x_next = descent_step(x)
        log_step(x_next)
        if should_terminate(x, x_next)
            return x_next
        end
        x = x_next
    end
    @warn "descent_until: max number of steps reached: max_steps $max_steps"
    x
end

abstract type FirstOrderMethod end

# Maximum Gradient Descent

struct MaximumGradientDescent <: FirstOrderMethod
    local_search
    term
    max_steps
    MaximumGradientDescent(;
        local_search=LineSearch(),
        term=TerminationTolerance(),
        max_steps=10_000) =
        new(local_search, term, max_steps)
end

function search(params::MaximumGradientDescent, f, ∇f, x_0; trace=false)
    descent_step = (x) -> descent_step_maxgd(params.local_search, f, ∇f, x)
    should_terminate = (x, x_next) -> check_termination(params.term, f, ∇f, x, x_next)
    if trace
        tl = TrainingLog(x_0, f(x_0), ∇f(x_0))
        log_step = (x) -> log_step!(tl, f, ∇f, x)
    else
        log_step = (x) -> nothing
    end
    x_opt = descent_until(descent_step, x_0, should_terminate, params.max_steps, log_step)
    x_opt, (trace ? tl : nothing)
end

function descent_step_maxgd(params::LineSearch, f, ∇f, x)
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
    should_terminate = (x, x_next) -> check_termination(params.term, f, ∇f, x, x_next)
    if trace
        tl = TrainingLog(x_0, f(x_0), ∇f(x_0))
        log_step = (x) -> log_step!(tl, f, ∇f, x)
    else
        log_step = (x) -> nothing
    end
    x_opt = descent_until(descent_step, x_0, should_terminate, params.max_steps, log_step)
    x_opt, (trace ? tl : nothing)
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
    should_terminate = (x, x_next) -> check_termination(params.term, f, ∇f, x, x_next)
    if trace
        tl = TrainingLog(x_0, f(x_0), ∇f(x_0))
        log_step = (x) -> log_step!(tl, f, ∇f, x)
    else
        log_step = (x) -> nothing
    end
    x_opt = descent_until(descent_step, x_0, should_terminate, params.max_steps, log_step)
    x_opt, (trace ? tl : nothing)
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
