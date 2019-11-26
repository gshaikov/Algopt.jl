module UniDerivative
"""
Compute derivative of a univariate function.
"""

abstract type DerivativeMethod end
struct CentralDiff <: DerivativeMethod end
struct ComplexDiff <: DerivativeMethod end

function df(::CentralDiff, f, x::Real; h = cbrt(eps()))::Real
    (f(x + h / 2) - f(x - h / 2)) / h
end

function df(::ComplexDiff, f, x::Real; h = 1e-20)::Real
    imag(f(x + h * im)) / h
end

end
