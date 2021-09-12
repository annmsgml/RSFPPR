#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <vector>
#include <algorithm>
using namespace std;

class Graph
{
private:
    int n, m, maxdeg;
    int **adj;
    int *data;
    int *deg;
public:
    Graph(/* args */);
    ~Graph();

    void readGraph_directed(const char *str);
    void writeGraph_directed(const char *str);

    void readGraph_undirected(const char *str);
    void writeGraph_undirected(const char *str);

    void readGraph_renumbering(const char *str);

    void readGraph_clq(const char *str);
    void writeGraph_clq(const char *str);

    void writeGraph_DirWithComma(const char *str);
};