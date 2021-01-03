using BenchmarkTools

import Algopt.LocalDescent

using Algopt.TestFunctions: Rosenbrock

ros = Rosenbrock(a = 1, b = 5)

ls = LocalDescent.LineSearch()
res = @btime LocalDescent.search_local(ls, ros.f, [-2, -2], [1, 1])
println("LineSearch: ", res)

sbt = LocalDescent.StrongBacktracking(σ=cbrt(eps()))
res = @btime LocalDescent.search_local(sbt, ros.f, ros.∇f, [-2, -2], [1, 1])
println("StrongBacktracking: ", res)
