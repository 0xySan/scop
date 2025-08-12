BIN_NAME := $(shell cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].name')

all : build

build:
	cargo build --release
	cp -f target/release/$(BIN_NAME) ./$(BIN_NAME)

test:
	cargo test

clean:
	cargo clean
	rm -f Cargo.lock

fclean: clean
	rm -f $(BIN_NAME)

check:
	cargo check

re: fclean build

.PHONY: all build test clean check re