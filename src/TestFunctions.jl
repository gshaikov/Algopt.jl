module TestFunctions

struct Rosenbrock
    f
    ∇f
    argmin
    Rosenbrock(; a=1, b=5) =
    new(x -> (a - x[1])^2 + b * (x[2] - x[1]^2)^2,
        x -> [
            -2 * (a - x[1]) - 4 * b * x[1] * (x[2] - x[1]^2),
            2 * b * (x[2] - x[1]^2)
        ],
        [a, a^2])
end

end # module
