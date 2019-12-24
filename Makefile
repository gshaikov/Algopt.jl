build:
	julia --project --color=yes src/Algopt.jl

tests: build
	julia --project --color=yes test/runtests.jl

benchmark-local_descent: _benchmark_local_descent
benchmark-first_order: _benchmark_first_order
_benchmark_%: build
	julia --project --color=yes benchmark/$*.jl

plot-first_order: _plot_first_order
_plot_%: build
	julia --project --color=yes -i plot/$*.jl
