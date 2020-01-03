using Plots

using Algopt.TestFunctions: Rosenbrock

using Algopt.FirstOrder

ros = Rosenbrock()

function solve_with_method(method)
    x_0 = [rand(-2:.1:2), rand(-2:.1:2)]
    res = FirstOrder.search(method, ros.f, ros.∇f, x_0, trace = true)
    println(res[:, end])
    println(size(res))

    x = y = range(-2, 2, step = .1)
    plot(contour(x, y, (x, y)->ros.f([x, y]); levels = range(0, 5, step = .1)))
    plot!(res[1,:], res[2,:], linewidth = 3)
end

function max_grad_descent()
    mgd = FirstOrder.MaximumGradientDescent()
    solve_with_method(mgd)
end

function grad_descent(α)
    grd = FirstOrder.GradientDescent(α = α)
    solve_with_method(grd)
end

function conjugate_grad_descent()
    cgd = FirstOrder.ConjugateGradientDescent()
    solve_with_method(cgd)
end
