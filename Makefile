ifeq ($(strip $(DEVKITARM)),)
$(error "Please set DEVKITARM in your environment. export DEVKITARM=<path to>/devkitARM")
endif

TOPDIR ?= $(CURDIR)
include $(DEVKITARM)/3ds_rules

TARGET          := Llama3DS
BUILD           := build
SOURCES         := source
INCLUDES        := include
APP_TITLE       := Llama3DS
APP_DESCRIPTION := Local AI Chat (llama2.c)
APP_AUTHOR      := Eko
APP_ICON        := $(TOPDIR)/resources/icon.png

ARCH            := -march=armv6k -mtune=mpcore -mfloat-abi=hard -mtp=soft
CFLAGS          := -g -Wall -O3 -mword-relocations -ffunction-sections $(ARCH) -D__3DS__
CXXFLAGS        := $(CFLAGS) -fno-rtti -fno-exceptions -std=gnu++17
ASFLAGS         := -g $(ARCH)
LDFLAGS         := -specs=3dsx.specs -g $(ARCH) -Wl,-Map,$(notdir $*.map)

LIBS            := -lcitro2d -lcitro3d -lctru -lm

LIBDIRS         := $(CTRULIB) $(PORTLIBS)

ifneq ($(BUILD),$(notdir $(CURDIR)))

export OUTPUT   := $(CURDIR)/$(TARGET)
export TOPDIR   := $(CURDIR)
export VPATH    := $(foreach dir,$(SOURCES),$(CURDIR)/$(dir))
export DEPSDIR  := $(CURDIR)/$(BUILD)

CFILES          := $(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.c)))
CPPFILES        := $(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.cpp)))
SFILES          := $(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.s)))

export LD       := $(CXX)
export OFILES_BIN :=
export OFILES_SRC := $(CPPFILES:.cpp=.o) $(CFILES:.c=.o) $(SFILES:.s=.o)
export OFILES     := $(OFILES_BIN) $(OFILES_SRC)
export HFILES_BIN :=

export INCLUDE  := $(foreach dir,$(INCLUDES),-I$(CURDIR)/$(dir)) \
                   $(foreach dir,$(LIBDIRS),-I$(dir)/include) \
                   -I$(CURDIR)/$(BUILD)

export LIBPATHS := $(foreach dir,$(LIBDIRS),-L$(dir)/lib)

export CFLAGS   += $(INCLUDE)
export CXXFLAGS += $(INCLUDE)

export _3DSXDEPS := $(OUTPUT).smdh

.PHONY: $(BUILD) clean all

all: $(BUILD)

$(BUILD):
	@[ -d $@ ] || mkdir -p $@
	@$(MAKE) --no-print-directory -C $(BUILD) -f $(CURDIR)/Makefile

clean:
	@echo clean ...
	@rm -fr $(BUILD) $(TARGET).3dsx $(TARGET).elf $(TARGET).smdh $(TARGET).cia

else

DEPENDS := $(OFILES:.o=.d)

$(OUTPUT).3dsx : $(OUTPUT).elf $(_3DSXDEPS)
$(OUTPUT).elf  : $(OFILES)

-include $(DEPENDS)

endif
