CC := gcc
CFLAGS := -g -fsanitize=address
LDFLAGS := -lblis -lm -fsanitize=address

main: main.o

clean:
	rm *.o main

.PHONY: clean
