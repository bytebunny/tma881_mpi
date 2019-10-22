CC := mpicc 
CFLAGS := -std=c11 -O3 -Wall -pthread -flto -ffast-math -march=native
LIBS := -lpthread -lm

OBJS := heat_diffusion.o 

.PHONY: all clean
all: heat_diffusion 

# Rule to generate object files:
heat_diffusion: $(OBJS) 
	$(CC) -o $@ $(OBJS) $(CFLAGS) $(LIBS)

$(OBJS) : helper.h

clean:
	rm -rvf *.o heat_diffusion 
