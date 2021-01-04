build:
	julia --project --color=yes src/Algopt.jl

tests: build
	julia --project --color=yes test/runtests.jl

arg ?= none

benchmark: build
	julia --project --color=yes benchmark/$(arg).jl

plot: build
	julia --project --color=yes -i plot/$(arg).jl
