#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <assert.h>
using namespace std;

class Graph
{
public:
    int n;                          // # of nodes
    long m;                          // # of edges
	int** inAdjList;
	int** outAdjList;
	int* indegree;
	int* outdegree;
	int** AdjList;
	int* degree;
    Graph()
    {
        n = m = 0;
    }

    ~Graph()
    {
    	for(int i = 0; i < n ;i++){
    		delete[] inAdjList[i];
    		delete[] outAdjList[i];
			delete[] AdjList[i];
    	}
        delete [] outAdjList;
        delete [] inAdjList;
		delete [] indegree;
		delete [] outdegree;
		delete [] AdjList;
		delete [] degree;
	}

    void inputGraph(string filename)
    {
    	m =0 ;
		// FILE* f = fopen(filename.c_str());
    	ifstream infile(filename.c_str());
    	infile >> n;
		// fread(&n, sizeof(int), 1, f);
		// fread(&m, sizeof(int), 1, f);
    	cout << "n= " << n << endl;
		// cout << "m= " << m << endl;
    	
		degree = new int[n];
		indegree = new int[n];
		outdegree = new int[n];
		for(int i = 0; i < n; i++){
			indegree[i] = 0;
			outdegree[i] = 0;
			degree[i] = 0;
		}
		int fromNode, toNode;
        int edgeCount = 0;
        while(infile >> fromNode >> toNode){
        	//infile >> fromNode >> toNode;
        	assert (toNode < n);
        	assert (fromNode < n);
        	outdegree[fromNode]++;
        	indegree[toNode]++;
			degree[fromNode]++;
			degree[toNode]++;
        }
        cout << "..." << endl;
        inAdjList = new int*[n];
        outAdjList = new int*[n];
		AdjList = new int*[n];
        int* pointer_in = new int[n];
        int* pointer_out = new int[n];
		int* pointer = new int[n];
        for(int i = 0; i < n; i++){
            pointer_out[i] = 0;
            pointer_in[i] = 0;
			pointer[i] = 0;
        }
        for(int i =0; i < n; i++){
            /*if(outdegree[i] == 0){
                outdegree[i] = 1;
                outAdjList[i] = new int[1];
                outAdjList[i][0] = i; 
                inAdjList[i] = new int[indegree[i] + 1];
                inAdjList[i][0] = i;
                pointer_out[i]++;
                pointer_in[i]++;
            }*/
            //else{
        	   inAdjList[i] = new int[indegree[i]];
        	   outAdjList[i] = new int[outdegree[i]];
			   AdjList[i] = new int[degree[i]];
            //}
        }
        
        infile.clear();
        infile.seekg(0);

        clock_t t0 = clock();
        infile >> n;
        cout << "n=: " << n << endl;
        while(infile >> fromNode >> toNode){
        	//infile >> fromNode >> toNode;
        	outAdjList[fromNode][pointer_out[fromNode]++] = toNode;
        	inAdjList[toNode][pointer_in[toNode]++] = fromNode;
			AdjList[fromNode][pointer[fromNode]++] = toNode;
			AdjList[toNode][pointer[toNode]++] = fromNode;
        	m++;
        	// cout << m << endl;
        }
        infile.close();
        clock_t t1 = clock();
        cout << "m = :" << m << endl;
        cout << "read file time: " << (t1 - t0) / (double) CLOCKS_PER_SEC << endl;
        delete[] pointer_in;
        delete[] pointer_out;
		delete[] pointer;
    }

	int getInSize(int vert){
		/*if(vert == 0)
			return inCount[0];
		else
			return inCount[vert] - inCount[vert - 1];*/
		return indegree[vert];
	}
	int getInVert(int vert, int pos){
		/*if(vert == 0)
			return inEdge[pos];
		else
			return inEdge[inCount[vert-1] + pos];*/
		return inAdjList[vert][pos];
	}
	int getOutSize(int vert){
		/*if(vert == 0)
			return outCount[0];
		else
			return outCount[vert] - outCount[vert - 1];*/
		return outdegree[vert];
	}
	int getOutVert(int vert, int pos){
		/*if(vert == 0)
			return outEdge[pos];
		else
			return outEdge[outCount[vert-1] + pos];*/
		return outAdjList[vert][pos];
	}
	int getDegree(int vert){
		return degree[vert];
	}
	int getNeighbor(int vert, int pos){
		return AdjList[vert][pos];
	}
	void toFile(string filename){
		ofstream output(filename);
		for(int i = 0; i < n; i++){
			for(int j = 0; j < outdegree[i]; j++){
				output << i << " " << outAdjList[i][j] << "\n";
			}
		}
		output.close();
	}
};


#endif
