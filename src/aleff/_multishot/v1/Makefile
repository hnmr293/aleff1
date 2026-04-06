PYTHON ?= python3
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
CC ?= cc
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O2 -fPIC
DEBUG_CFLAGS = -std=c2x -Wall -Wextra -Wpedantic -O0 -g3 -fPIC

TARGET = _aleff$(EXT_SUFFIX)
SRC = _aleff.c

.PHONY: build debug clean

build: $(TARGET)

debug: $(SRC)
	$(CC) $(DEBUG_CFLAGS) -shared -I$(INCLUDE) -o $(TARGET) $<

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -shared -I$(INCLUDE) -o $@ $<

clean:
	rm -f _aleff*.so
