NAME := tests
CC := c++
CFLAGS := -Wall -Werror -Wextra -I. -I../includes -std=c++17
HEADERS := doctest.h ../includes/Tensor.hpp ../includes/Layout.hpp ../includes/TensorView.hpp

SRC_DIR := src
SRC := $(wildcard $(SRC_DIR)/*.cpp)

all: $(NAME)

$(NAME): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(SRC) -o $(NAME)

run: $(NAME)
	./$(NAME)

clean:
	@rm -f $(NAME)

fclean: clean

re: fclean all

.PHONY: all run clean fclean re
