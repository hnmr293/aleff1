PYTHON ?= python3
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
LDFLAGS := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LDSHARED').split(' ', 1)[1] if sysconfig.get_config_var('LDSHARED') else '')")
CC ?= cc
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O2 -fPIC

TARGET = src/aleff/_aleff$(EXT_SUFFIX)
SRC = src/aleff/_aleff.c

.PHONY: build clean install test

build: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -shared -I$(INCLUDE) -o $@ $<

clean:
	rm -f $(TARGET)

install:
	uv pip install -e ".[dev]"

test: build
	uv run pytest tests/ -v
