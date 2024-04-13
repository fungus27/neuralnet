CC := gcc
CFLAGS := -DGLEW_NO_GLU -g
LDFLAGS := -lm -lblis -lGL -lglfw -lGLEW -g

interactive: interactive.o ../neural_network.o
	$(CC) $(LDFLAGS) $(CFLAGS) interactive.o ../neural_network.o ../activation.o ../loss.o -o interactive

clean:
	rm interactive *.o

.PHONY: clean
