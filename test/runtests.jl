tests = [
    "uni_derivative.jl",
    "bracketing.jl",
    "local_descent.jl",
    "first_order.jl",
]

for test in tests
    include(test)
end
