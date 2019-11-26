module UniDerivative
"""
Compute derivative of a univariate function.
"""

abstract type DerivativeMethod end

struct CentralDiff <: DerivativeMethod
    h
    CentralDiff(h = cbrt(eps())) = new(h)
end

function df(param::CentralDiff, f, x::Real)::Real
    (f(x + param.h / 2) - f(x - param.h / 2)) / param.h
end

struct ComplexDiff <: DerivativeMethod
    h
    ComplexDiff(h = 1e-20) = new(h)
end

function df(param::ComplexDiff, f, x::Real)::Real
    imag(f(x + param.h * im)) / param.h
end

end
