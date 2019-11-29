module Benchmark

using BenchmarkTools

import Algopt.LocalDescent
quad0 = x->x'x
∇quad0 = x->2x
@btime LocalDescent.search(LocalDescent.LineSearch(), quad0, [10, 10], [-1, -1])
@btime LocalDescent.search(LocalDescent.StrongBacktracking(), quad0, ∇quad0, [10, 10], [-1, -1])

import Algopt.FirstOrder
using Algopt.TestFunctions: Rosenbrock
ros = Rosenbrock()
mgd = FirstOrder.MaximumGradientDescent()
term = FirstOrder.Termination(max_iter = 1e6, ϵ_abs = 1e-6, ϵ_rel = 1e-6, ϵ_grad = 1e-6)
@btime FirstOrder.search($mgd, $ros.f, $ros.∇f, x_0, term = $term) setup = (x_0 = [rand(-2:.1:2), rand(-2:.1:2)])

end
