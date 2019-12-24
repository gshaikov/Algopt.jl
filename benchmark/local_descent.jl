using BenchmarkTools

import Algopt.LocalDescent

quad0 = x->x'x
∇quad0 = x->2x

ls = LocalDescent.LineSearch()
res = @btime LocalDescent.search(ls, quad0, [10, 10], [-1, -1])
println("LineSearch: ", res)

sbt = LocalDescent.StrongBacktracking()
res = @btime LocalDescent.search(sbt, quad0, ∇quad0, [10, 10], [-1, -1])
println("StrongBacktracking: ", res)
