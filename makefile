OBJS = main.o
CC = g++
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG)

all : $(OBJS)
	$(CC) $(OBJS) -o array

main.o : main.cpp array.h
	$(CC) $(CFLAGS) main.cpp

clean :
	\rm *.o