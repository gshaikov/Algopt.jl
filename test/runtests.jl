module AlgoptTests

import Algopt

# Make sure to not start with an outdated registry
# https://github.com/JuliaLang/Pkg.jl/blob/master/test/runtests.jl
rm(joinpath(@__DIR__, "registries"); force=true, recursive=true)

tests = [
    "uni_derivative.jl",
    "bracketing.jl",
    "local_descent.jl",
    "first_order.jl",
    "test_functions.jl",
]

for test in tests
    include(test)
end

# clean up locally cached registry
# https://github.com/JuliaLang/Pkg.jl/blob/master/test/runtests.jl
rm(joinpath(@__DIR__, "registries"); force=true, recursive=true)

end # module
