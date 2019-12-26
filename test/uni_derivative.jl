module UniDerivativeTests

import ..Algopt # ensure we are using the correct Algopt

using Test

using Algopt.UniDerivative:
df,
CentralDiff, ComplexDiff

quadratic = x->x^2

@testset "Algopt.UniDerivative.df(::CentralDiff)" begin
    @test df(CentralDiff(), quadratic, 3) ≈ 2 * 3
end

@testset "Algopt.UniDerivative.df(::ComplexDiff)" begin
    @test df(ComplexDiff(), quadratic, 3) ≈ 2 * 3
end

end # module
