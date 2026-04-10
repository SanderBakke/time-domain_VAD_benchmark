# Makefile for embedded/proto_energy — host build
#
# Builds and runs the energy kernel test against Python-generated test vectors.
# This is for local (laptop) verification before moving to the devkit.
#
# Usage:
#   make              # build
#   make run          # build and run
#   make clean        # remove build artifacts
#
# Devkit cross-compile (template — adjust prefix and BSP flags for your target):
#   riscv64-unknown-elf-gcc -O0 -Wall -std=c99 -I../test_vectors \
#       -DPLATFORM_DEVKIT \
#       [your BSP includes and link flags here] \
#       -o test_energy.elf test_energy.c energy_kernel.c

CC      = gcc
CFLAGS  = -O0 -Wall -Wextra -std=c99 -I../test_vectors
TARGET  = test_energy
SRCS    = test_energy.c energy_kernel.c

TESTVEC = ../test_vectors/test_vectors.h

.PHONY: all run clean check_testvec

all: check_testvec $(TARGET)

$(TARGET): $(SRCS) $(TESTVEC) energy_kernel.h platform.h
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

run: all
	./$(TARGET)

check_testvec:
	@if [ ! -f $(TESTVEC) ]; then \
		echo ""; \
		echo "ERROR: test_vectors.h not found."; \
		echo "Generate it first by running (from the repo root):"; \
		echo "  python embedded/proto_energy/ref/export_testvec.py"; \
		echo ""; \
		exit 1; \
	fi

clean:
	rm -f $(TARGET)
