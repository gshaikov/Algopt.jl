build:
	julia --project=dev --color=yes src/Algopt.jl

tests: build
	julia --project=dev --color=yes test/runtests.jl

benchmark: build
	julia --project=dev --color=yes test/benchmarking.jl

plot-first-order: _plot_first_order
_plot_%:
	julia --project=dev --color=yes -i plot/$*.jl
