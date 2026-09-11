NAME := strata
CC := c++
CFLAGS := -Wall -Werror -Wextra -Iincludes -std=c++17

test:
	$(MAKE) -C test run

.PHONY: test

