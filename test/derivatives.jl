test_func = x->x^2

@testset "derivatives.jl df" begin
    x = 3
    result = Algopt.Derivatives.df(test_func, x)
    @test result == 2x
    result = Algopt.Derivatives.df(test_func, x; method = Algopt.Derivatives.complex)
    @test result == 2x
end

@testset "derivatives.jl df_complex" begin
    x = 3
    result = Algopt.Derivatives.df_complex(test_func, x)
    @test result == 2x
end
