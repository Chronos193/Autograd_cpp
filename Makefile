CXX = g++
CXXFLAGS = -std=c++17 -Wall -Iinclude

SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))
TARGET = $(BINDIR)/my_autograd.exe
EXTRA_DOT = my_network.dot 
EXTRA_PNG = my_network.png

$(TARGET): $(OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BINDIR):
	-mkdir $(BINDIR)

$(OBJDIR):
	-mkdir $(OBJDIR)

.PHONY: clean clean-all png
 
clean:
	-rmdir /S /Q $(OBJDIR) 2>nul
	-rmdir /S /Q $(BINDIR) 2>nul

clean-all: clean
	-del /Q $(EXTRA_DOT) $(EXTRA_PNG) 2>nul

png:
	dot -Tpng $(EXTRA_DOT) -o $(EXTRA_PNG)
