module TestUniDerivative

using Test
using Algopt.UniDerivative: df, df_complex

quadratic = x->x^2

@testset "uni_derivative.jl df" begin
    @test df(quadratic, 3) == 2 * 3
end

@testset "uni_derivative.jl df_complex" begin
    @test df_complex(quadratic, 3) == 2 * 3
end

end
