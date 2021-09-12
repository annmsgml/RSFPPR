all: processgh
.PHONY : all

CC = g++ "-std=c++11"
CONFIG = -O3 -g
#objects = uncertain-core*.o graph.o

processgraph_obj = ProcessGraphs.o

processgh: $(processgraph_obj)
	$(CC) -o processgh $(processgraph_obj) $(CONFIG)

$(processgraph_obj):%.o:%.cpp
	$(CC) -c $< $(CONFIG) -o $@

#%.o:%.c
#	$(CC) -c $^
.PHONY : clean
clean:
	rm -f *.o processgh