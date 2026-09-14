NAME := strata
CC := c++
CFLAGS := -Wall -Werror -Wextra -Iincludes -std=c++17

test:
	$(MAKE) -C test run

clean:
	$(MAKE) -C test clean

fclean:
	$(MAKE) -C test fclean

.PHONY: fclean clean test
