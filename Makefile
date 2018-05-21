CROSSTOOL = aarch64-linux-android
CC = $(CROSSTOOL)-gcc
AR = $(CROSSTOOL)-ar

SRCFILE := $(wildcard ./src/*.c)
SRCFILE += $(wildcard ./src/arm64/*.S)

OBJFILE := $(SRCFILE)

CFLAG 	:= -I. -Iinclude -fopenmp -std=c99 -DSTAND_ALONE_COMPILER
ASMFLAG := -c -I. -Iinclude

AllObjFile = $(SRCFILE:.c=.o)

all:$(AllObjFile)
	$(CC) -static -O2 -I./include ./test/sgemm-test.cpp $(AllObjFile) -pthread -lm -lgomp -o sgemm-test-android

%.o: %.c
	$(CC) $(CFLAG) -o $@ -c $<
%.o: %.s
	$(CC) $(ASMFLAG) -o $@ -c $<
	
