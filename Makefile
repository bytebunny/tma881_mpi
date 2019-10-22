CC := gcc
CFLAGS := -std=c11 -O3 -Wall -flto -march=native -ffast-math
LIBS := -lm

OBJS := heat_diffusion.o 

.PHONY: all test clean
all: heat_diffusion 

# Rule to generate object files:
heat_diffusion: $(OBJS) 
	$(CC) -o $@ $(OBJS) $(CFLAGS) $(LIBS)

$(OBJS) : heat_diffusion.h

test:
	tar -czvf heat_diffusion.tar.gz heat_diffusion.c heat_diffusion.h Makefile
	./check_submission.py heat_diffusion.tar.gz
clean:
	rm -rvf *.o heat_diffusion extracted/ heat_diffusion.tar.gz vgcore*
