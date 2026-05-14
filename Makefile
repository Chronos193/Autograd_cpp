CXX = g++
CXXFLAGS = -std=c++17 -Wall

OBJS = main.o Value.o Visualize.o
TARGET = my_autograd.exe
EXTRA_DOT = my_network.dot 
EXTRA_PNG = my_network.png

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean clean-all png
 
clean:
	del /Q $(OBJS) $(TARGET) 2>nul

clean-all:
	del /Q $(OBJS) $(TARGET) $(EXTRA_DOT) $(EXTRA_PNG) 2>nul

png:
	dot -Tpng $(EXTRA_DOT) -o $(EXTRA_PNG)
