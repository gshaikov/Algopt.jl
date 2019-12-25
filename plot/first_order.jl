using Plots

using Algopt.TestFunctions: Rosenbrock

using Algopt.FirstOrder

ros = Rosenbrock()
term = FirstOrder.Termination(max_iter = 1e6, ϵ_abs = 1e-6, ϵ_rel = 1e-6, ϵ_grad = 1e-6)

function max_grad_descent()
    mgd = FirstOrder.MaximumGradientDescent()

    x_0 = [rand(-2:.1:2), rand(-2:.1:2)]
    res = FirstOrder.search(mgd, ros.f, ros.∇f, x_0, term = term, trace = true)
    println(res[:, end])
    println(size(res))

    x = y = range(-2, 2, step = .1)
    plot(contour(x, y, (x, y)->ros.f([x, y]); levels = range(0, 5, step = .1)))
    plot!(res[1,:], res[2,:], linewidth = 3)
end

function grad_descent(α)
    gd = FirstOrder.GradientDescent(α = α)

    x_0 = [rand(-2:.1:2), rand(-2:.1:2)]
    res = FirstOrder.search(gd, ros.f, ros.∇f, x_0, term = term, trace = true)
    println(res[:, end])
    println(size(res))

    x = y = range(-2, 2, step = .1)
    plot(contour(x, y, (x, y)->ros.f([x, y]); levels = range(0, 5, step = .1)))
    plot!(res[1,:], res[2,:], linewidth = 3)
end

function conjugate_grad_descent()
    cgd = FirstOrder.ConjugateGradientDescent()

    x_0 = [rand(-2:.1:2), rand(-2:.1:2)]
    res = FirstOrder.search(cgd, ros.f, ros.∇f, x_0, term = term, trace = true)
    println(res[:, end])
    println(size(res))

    x = y = range(-2, 2, step = .1)
    plot(contour(x, y, (x, y)->ros.f([x, y]); levels = range(0, 5, step = .1)))
    plot!(res[1,:], res[2,:], linewidth = 3)
end
