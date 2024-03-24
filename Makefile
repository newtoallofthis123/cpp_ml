build:
	@g++ -std=c++20 -o ./bin/main main.cpp
run: build
	@./bin/main
