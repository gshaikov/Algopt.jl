tests = [
    "uni_derivative.jl",
    "bracketing.jl",
    "local_descent.jl",
]

for test in tests
    include(test)
end
