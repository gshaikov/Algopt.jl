tests = [
    "uni_derivative.jl",
    "bracketing.jl",
]

for test in tests
    include(test)
end
