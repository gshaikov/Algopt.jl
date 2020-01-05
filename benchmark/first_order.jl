using BenchmarkTools

import Algopt.FirstOrder

using Algopt.TestFunctions: Rosenbrock

ros = Rosenbrock(a = 1, b = 5)

mgd = FirstOrder.MaximumGradientDescent()
res = @btime FirstOrder.search($mgd, $ros.f, $ros.∇f, x_0) setup = (x_0 = [rand(-2:.1:2), rand(-2:.1:2)])
println("MaximumGradientDescent: ", res)

grd = FirstOrder.GradientDescent()
term = FirstOrder.Termination(max_iter = 100_000)
res = @btime FirstOrder.search($grd, $ros.f, $ros.∇f, x_0; term = $term) setup = (x_0 = [rand(-2:.1:2), rand(-2:.1:2)])
println("GradientDescent: ", res)

cgd = FirstOrder.ConjugateGradientDescent()
res = @btime FirstOrder.search($cgd, $ros.f, $ros.∇f, x_0) setup = (x_0 = [rand(-2:.1:2), rand(-2:.1:2)])
println("ConjugateGradientDescent: ", res)
