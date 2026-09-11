NAME := tests
CC := c++
CFLAGS := -Wall -Werror -Wextra -I. -I../includes -std=c++17
DOCTEST_FLAGS := -DDOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

SRC_DIR := src
SRC := $(wildcard $(SRC_DIR)/*.cpp)

all: $(NAME)

$(NAME): $(SRC) doctest.h ../includes/Tensor.hpp
	$(CC) $(CFLAGS) $(DOCTEST_FLAGS) $(SRC) -o $(NAME)

run: $(NAME)
	./$(NAME)

clean:
	@rm -f $(NAME)

fclean: clean

re: fclean all

.PHONY: all run clean fclean re
