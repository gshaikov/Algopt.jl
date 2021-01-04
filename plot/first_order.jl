using Plots

using LinearAlgebra

using Algopt.TestFunctions: Rosenbrock

using Algopt.FirstOrder

ros = Rosenbrock()

function solve_with_method(method)
    x_0 = [rand(-2:.1:2), rand(-2:.1:2)]
    println("Start point: ", x_0)
    res, train_log = FirstOrder.search(method, ros.f, ros.∇f, x_0, trace=true)
    println("Optimal: ", res)
    n_steps = size(train_log.x)[2]
    println("Steps: ", n_steps)

    dfx = train_log.fx[1,end - 1] - train_log.fx[1,end]
    abs_err = dfx
    rel_err = dfx / abs(train_log.fx[1,end - 1])
    grad_err = norm(train_log.∇fx[:,end])
    println("abs_err:  $abs_err")
    println("rel_err:  $rel_err")
    println("grad_err: $grad_err")

    println("Last two x: $(train_log.x[:, (end - 1):end])")

    x = y = range(-2, 2, step=.1)
    steps_plot = plot(contour(x, y, (x, y) -> ros.f([x, y]); levels=range(0, 5, step=.1)))
    steps_plot = plot!(train_log.x[1,:], train_log.x[2,:], linewidth=3)

    cost_plot = plot(1:n_steps, train_log.fx[1,:], title="f(x)")

    plot(steps_plot, cost_plot, layout=(1, 2))
end

function max_grad_descent()
    mgd = FirstOrder.MaximumGradientDescent()
    solve_with_method(mgd)
end

function grad_descent(α)
    grd = FirstOrder.GradientDescent(α=α)
    solve_with_method(grd)
end

function conjugate_grad_descent()
    cgd = FirstOrder.ConjugateGradientDescent()
    solve_with_method(cgd)
end
