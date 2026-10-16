NAME := strata
CC := c++
CFLAGS := -Wall -Wextra -Werror -Iincludes -std=c++17

test:
	$(MAKE) -C test run

clean:
	$(MAKE) -C test clean

fclean:
	$(MAKE) -C test fclean

.PHONY: fclean clean test
