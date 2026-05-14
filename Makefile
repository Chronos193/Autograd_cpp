CXX = g++
CXXFLAGS = -std=c++17 -Wall

OBJS = main.o Value.o Visualize.o
TARGET = my_autograd.exe
EXTRA = my_network.dot my_network.png

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean clean-all
 
clean:
	del /Q $(OBJS) $(TARGET) 2>nul

clean-all:
	del /Q $(OBJS) $(TARGET) $(EXTRA) 2>nul
