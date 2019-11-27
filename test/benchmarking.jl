module BenchmarkAlgopt

using BenchmarkTools

using Algopt.LocalDescent:
search,
LineSearch, StrongBacktracking


quad0 = x->x'x
∇quad0 = x->2x

@btime search(LineSearch(), quad0, [10, 10], [-1, -1])
@btime search(StrongBacktracking(), quad0, ∇quad0, [10, 10], [-1, -1])

end