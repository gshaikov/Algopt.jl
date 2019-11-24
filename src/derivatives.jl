module Derivatives

@enum DiffMethod begin
    complex = 1
end

function df(f, x; method = complex)
    if method == complex
        df_complex(f, x)
    end
end

function df_complex(f, x; h = 1e-20)
    imag(f(x + h * im)) / h
end

end
