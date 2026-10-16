test:
	$(MAKE) -C test run

clean:
	$(MAKE) -C test clean

fclean:
	$(MAKE) -C test fclean

.PHONY: fclean clean test
