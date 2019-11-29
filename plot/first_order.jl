using Plots

using Algopt.TestFunctions: Rosenbrock

using Algopt.FirstOrder

function max_grad_descent()
    ros = Rosenbrock()
    mgd = FirstOrder.MaximumGradientDescent()
    term = FirstOrder.Termination(max_iter = 1e6, ϵ_abs = 1e-6, ϵ_rel = 1e-6, ϵ_grad = 1e-6)

    x_0 = [rand(-2:.1:2), rand(-2:.1:2)]
    res = FirstOrder.search(mgd, ros.f, ros.∇f, x_0, term = term, trace = true)
    println(res[:, end])

    x = y = range(-2, 2, step = .1)
    plot(contour(x, y, (x, y)->ros.f([x, y]); levels = range(0, 5, step = .1)))
    plot!(res[1,:], res[2,:], linewidth = 3)
end
