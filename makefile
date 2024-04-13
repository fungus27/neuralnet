CC := gcc
CFLAGS := -O3
LDFLAGS := -lblis -lm -O3

main: main.o neural_network.o loss.o activation.o optimizer.o
main.o: neural_network.h loss.h activation.h optimizer.h

neural_newtork.o: neural_newtork.h loss.h activation.h

loss.o: loss.h

activation.o: activation.h

optimizer.o: optimizer.h neural_network.h

clean:
	rm *.o main

.PHONY: clean
