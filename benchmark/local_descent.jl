using BenchmarkTools

import Algopt.LocalDescent

using Algopt.TestFunctions: Rosenbrock

ros = Rosenbrock(a = 1, b = 5)

ls = LocalDescent.LineSearch()
res = @btime LocalDescent.search(ls, ros.f, [-2, -2], [1, 1])
println("LineSearch: ", res)

sbt = LocalDescent.StrongBacktracking()
res = @btime LocalDescent.search(sbt, ros.f, ros.∇f, [-2, -2], [1, 1])
println("StrongBacktracking: ", res)
