# Makefile for PinePhone Pro audio denoiser tools
 
SHERPA_PREFIX := $(HOME)/pp_voice_agent/.venv/lib/python3.14/site-packages/sherpa_onnx
 
CC      := gcc
CFLAGS  := -O2 -Wall -Wextra \
           $(shell pkg-config --cflags libpipewire-0.3) \
           -I$(SHERPA_PREFIX)/include
LDFLAGS_COMMON := -L$(SHERPA_PREFIX)/lib \
           -lsherpa-onnx-c-api -lonnxruntime \
           -Wl,-rpath,$(SHERPA_PREFIX)/lib
 
.PHONY: all clean
 
all: denoise_source denoise_stream
 
denoise_source: denoise_source.c
	$(CC) $(CFLAGS) -o $@ $< \
		$(shell pkg-config --libs libpipewire-0.3) \
		-lasound -lpthread \
		$(LDFLAGS_COMMON)
 
denoise_stream: denoise_stream.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS_COMMON)
 
clean:
	rm -f denoise_source denoise_stream
