using BenchmarkTools

import Algopt.FirstOrder
using Algopt.TestFunctions: Rosenbrock

ros = Rosenbrock()

term = FirstOrder.Termination(max_iter = 1e6, ϵ_abs = 1e-6, ϵ_rel = 1e-6, ϵ_grad = 1e-6)

mgd = FirstOrder.MaximumGradientDescent()
res = @btime FirstOrder.search($mgd, $ros.f, $ros.∇f, x_0, term = $term) setup = (x_0 = [rand(-2:.1:2), rand(-2:.1:2)])
println("MaximumGradientDescent: ", res)

gd = FirstOrder.GradientDescent(α = 1e-2)
res = @btime FirstOrder.search($gd, $ros.f, $ros.∇f, x_0, term = $term) setup = (x_0 = [rand(-2:.1:2), rand(-2:.1:2)])
println("GradientDescent: ", res)
