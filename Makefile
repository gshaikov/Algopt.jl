build:
	julia --project=. --color=yes src/Algopt.jl

tests: build
	julia --project=. --color=yes test/runtests.jl

benchmark: build
	julia --project=. --color=yes test/benchmarking.jl
