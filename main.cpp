#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "ppr.h"
#include <unordered_set>    
#include <cstdlib>
#include <cstring>

void usage() {
    cerr << "Wrong Parameters" << endl;
}

int check_inc(int i, int max) {
    if (i == max) {
        usage();
        exit(1);
    }
    return i + 1;
}

bool maxCmp(const pair<int, double>& a, const pair<int, double>& b){
    return a.second > b.second;
}

vector<int> getRealTopK(int s, int k, string target_filename, int vert){
    stringstream ss;
    ss << "ppr-answer/" << target_filename << "/" << s << ".txt";
    string infile = ss.str();
    ifstream real(infile);
    vector<int> realList;
    vector<double> simList;
    for(int i = 0; i < vert; i++){
        int tempId; 
        double tempSim;
        real >> tempId >> tempSim;
        if(i >= k && tempSim < simList[k-1]){
           break; 
        } 
        realList.push_back(tempId);
        simList.push_back(tempSim);
    }
    real.close();
    return realList;
}

unordered_map<int, double> getRealTopKMap(int s, int k, string target_filename, int vert){
    unordered_map<int, double> answer_map;
    stringstream ss;
    ss << "ppr-answer/" << target_filename << "/" << s << ".txt";
    string infile = ss.str();
    ifstream real(infile);
    double k_Sim = 0;
    for(int i = 0; i < vert; i++){
        int tempId;
        double tempSim;
        real >> tempId >> tempSim;
        if(i == k - 1){
            k_Sim = tempSim;
        }
        if(i >= k && tempSim < k_Sim){
            break;
        }
        answer_map[tempId] = tempSim;
    }
    real.close();
    return answer_map;
}

int main(int argc, char *argv[]){
    int i = 1;
    char *endptr;
    string filename;
    double alpha = 0.2;            //decay factor
    int node_count = 20;           //query node size
    string algo = "RSFPPR";
    int algorithm_option = 4;
    if(argc < 1){
        usage();
        exit(1);
    }
    while (i < argc) {
        if (!strcmp(argv[i], "-d")) {
            i = check_inc(i, argc);
            filename = argv[i];
        } 
        else if (!strcmp(argv[i], "-algo")) {
            i = check_inc(i, argc);
            algo = argv[i];
        }
        else if (!strcmp(argv[i], "-o")) {
            i = check_inc(i, argc);
            algorithm_option = strtod(argv[i], &endptr);
            cout << "option: " << algorithm_option << endl;
        }
        else if (!strcmp(argv[i], "-n")) {
            i = check_inc(i, argc);
            node_count = strtod(argv[i], &endptr);
            if ((node_count < 0) && endptr) {
                cerr << "Invalid node_count argument" << endl;
                exit(1);
            }
        }
        // else if (!strcmp(argv[i], "-r")) {
        //     i = check_inc(i, argc);
        //     error_rate = strtod(argv[i], &endptr);
        //     if (((error_rate < 0) || (error_rate > 1)) && endptr) {
        //         cerr << "Invalid error_rate argument" << endl;
        //         exit(1);
        //     }
        // }
        // else if (!strcmp(argv[i], "-err")) {
        //     i = check_inc(i, argc);
        //     error_eps = strtod(argv[i], &endptr);
        //     if (((error_eps < 0) || (error_eps > 1)) && endptr) {
        //         cerr << "Invalid error_eps argument" << endl;
        //         exit(1);
        //     }
        // }
        else if (!strcmp(argv[i], "-a")) {
            i = check_inc(i, argc);
            alpha = strtod(argv[i], &endptr);
            if (((alpha < 0) || (alpha > 1)) && endptr) {
                cerr << "Invalid alpha argument" << endl;
                exit(1);
            }
        }
        else {
            usage();
            exit(1);
        }
        i++;
    }
    
    PPR ppr = PPR(filename, alpha);
    if(algo == "GEN_QUERY"){
        ofstream outFile("dataset/" + filename + ".query");
        ppr.generateQueryNode(node_count, outFile);
        outFile.close(); 
    }
    else if(algo == "GEN_GROUND_TRUTH"){
        string queryname = "dataset/" + filename + ".query";
        if(!ppr.is_file_exist(queryname)){
            cout << "please generate query file first" << endl;
        }
        else {
            ppr.PowerMethodMulti(10000, node_count, 10);/*  多线程PowerMethparameter: iteration loops, node size, thread num */   
        }
    }
    else if (algo == "RSFPPR") {
        cout << "Algorithm for personalized Pagerank with loop" << endl;
        string queryname = "dataset/" + filename + ".query";
        if(!ppr.is_file_exist(queryname)){
            cout << "please generate query file first" << endl;
        }
        else if(algorithm_option == 1){
            // cout << "already generate groundtruth file" << endl;
            ppr.PowerMethodMulti(2000, node_count, 10); 
            cout << "avg conductance: " << ppr.avg_conductance / (double) node_count << endl;
            // cout << "avg clustersize: " << ppr.avg_clustersize / (double) node_count << endl;
            ppr.avg_conductance = 0;   
            ppr.avg_clustersize = 0;
        } else if(algorithm_option == 2) {
            ppr.MC_algorithm2(node_count);
            cout << "avg absolute error: " << ppr.avg_abs_err / (double) node_count << endl;
            cout << "avg l1 error: " << ppr.avg_L1_error / (double) node_count << endl;
            cout << "avg running time: " << ppr.avg_time / (double) node_count << endl;
            cout << "avg conductance: " << ppr.avg_conductance / (double) node_count << endl;
            // cout << "avg clustersize: " << ppr.avg_clustersize / (double) node_count << endl;

            cout << "----------------------------------------------------------------------" << endl;
            cout << "Compare estimator: Simple walk" << endl;
            cout << "avg absolute error: " << ppr.avg_abs_err_Compare / (double) node_count << endl;
            cout << "avg l1 error: " << ppr.avg_L1_error_Compare / (double) node_count << endl;

        } else if(algorithm_option == 3) {
            ppr.push(node_count);
            cout << "avg absolute error: " << ppr.avg_abs_err / (double) node_count << endl;
            cout << "avg l1 error: " << ppr.avg_L1_error / (double) node_count << endl;
            cout << "avg running time: " << ppr.avg_time / (double) node_count << endl;
            cout << "avg conductance: " << ppr.avg_conductance / (double) node_count << endl;
            // cout << "avg clustersize: " << ppr.avg_clustersize / (double) node_count << endl;
        } else if(algorithm_option == 4) {
            ppr.comparetime();
        }

    } else {
        cout << "no algorithm input" << endl;
    }
    return 0;
};
