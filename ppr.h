#ifndef PPR_H
#define PPR_H

#include <vector>
#include <algorithm>
#include <queue>
#include <functional>
#include <iostream>
#include <fstream>
#include <future>
#include <string>
#include <sstream>
#include "Graph.h"
#include "Random.h"
#include "alias.h"
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <sys/time.h> 
#include <time.h>
#include <cstring>

double alpha = 0.01;
double q = 1e-2;

double c = 1;
double error_threshold = 0.0000000001;

bool maxScoreCmp(const pair<int, double>& a, const pair<int, double>& b){
    return a.second > b.second;
}

class pqcompare
{
  bool reverse;
public:
  pqcompare(const bool& revparam=false)
    {reverse=revparam;}
  bool operator() (const pair<int, double>& lhs, const pair<int, double>&rhs) const
  {
    if (reverse) return (lhs.second > rhs.second);
    else return (lhs.second < rhs.second);
  }
};


void RandomWalk(int walk_num, Alias &alias, Random &R, Graph& g, int* vert_count){
    for(int i = 0; i < walk_num; i++){
        int tempNode = alias.generateRandom_t(R);
        vert_count[tempNode]++;
        while(R.drand_t() > 0.2){
            int length = g.getOutSize(tempNode);
            if(length > 0){   
                int r = R.generateRandom_t() % length;
                tempNode = g.getOutVert(tempNode, r);
            }
            vert_count[tempNode]++;
        }
    }
}

class PPR
{
friend void ppr_t_PowerMethod(PPR* ppr, vector<int> nodeList, int iterations);
public:
    double avg_pre;
    double avg_recall;
    double avg_time;
    double avg_NDCG;
    double avg_conductance;
    double avg_clustersize;
    double threshold_num;
    int error_num;
    int k;
    double error_rate;
    double error_eps;
    double bound_c;
    double forward_c;
    int tempAvgInterval;
    int maxInterval;
    Graph g;
    Random R;
    int vert;
    double alpha;
    string target_filename;
    double* pow_alpha;
    double* vert_count;
    int value_count;
    int* value_verts;
    double* r_t;
    double* r; 
    double* rmap_back;
    bool* isInQueueBack;
    unsigned NUM_CORES;
    int** multiVertCount;
    double* resultList;
    double back_walk_num;
    double avg_L1_error;
    double avg_max_error;
    double max_max_error;
    double avg_avg_error;
    double avg_top_err[500] = {0};
    double avg_abs_err;
    double avg_abs_err_Compare;
    double avg_L1_error_Compare;
    int* r_hash_arr;
    bool* r_item_arr;
    double* r_max_c;
    double avg_forward_time;
    double avg_rsum;
    double* q_vector;
    vector<vector<pair<int, double> > > r_vec;
    void PowerMethodMulti(int iterations, int node_count, int num_thread);
    // void PowerMethodMulti_abs(int iterations, int node_count, int num_thread);
    // const static int NUMTHREAD = 20;
    Random* Rs;

    PPR(string name, double input_alpha) {
        // k = k_num;
        // error_rate = e_rate;
        // error_eps = e_eps;
        // if(error_rate != 1){
        //     bound_c = 0.2;
        //     forward_c = 0.2;
        // }
        // else{
        //     bound_c = 1.7;
        //     //bound_c = 0.5;
        //     forward_c = 1;
        // }
        error_num = k * (1-error_rate);
        avg_L1_error = 0;
        avg_max_error = 0;
        max_max_error = 0;
        avg_avg_error = 0;
        avg_time = 0;
        avg_conductance = 0;
        avg_pre = 0 ;
        avg_recall = 0;
        avg_NDCG = 0;
        avg_rsum = 0;
        avg_forward_time = 0;
        avg_clustersize = 0;
        tempAvgInterval = 0;
        maxInterval = 0;
        back_walk_num = 0;
        threshold_num = 0;
        target_filename = name;
        string filename = "dataset/" + name + ".txt";
        g.inputGraph(filename);
        cout << "edge num: " << g.m << endl;
        vert = g.n;
        alpha = input_alpha;
        srand(unsigned(time(0)));
        R = Random(unsigned(rand()));
        vert_count = new double[vert];
        resultList = new double[vert];
        value_count = 0;
        r = new double[vert];
        rmap_back = new double[vert];
        value_verts = new int[vert];
        isInQueueBack = new bool[vert];
        r_hash_arr = new int[vert];
        r_item_arr = new bool[vert];
        r_max_c = new double[vert];
        q_vector = new double[vert];
        for(int i =0 ; i < vert; i++){
            resultList[i] = 0;
            vert_count[i] = 0;
            value_verts[i] = -1;
            r[i] = 0;
            vector<pair<int, double> > temp_r_vec;
            r_vec.push_back(temp_r_vec);
            isInQueueBack[i] = false;
            rmap_back[i] = 0;
            r_hash_arr[i] = 0;
            r_item_arr[i] = false;
            r_max_c[i] = g.getInSize(i);
            //r_max_c[i] = 1;

            q_vector[i] = q/(1-q)*g.getDegree(i);
            // q_vector[i] = q*g.getDegree(i);
            // q_vector[i] = q;
        }
        NUM_CORES = std::thread::hardware_concurrency();
        assert(NUM_CORES >= 2);
        cout << "thread core: " << NUM_CORES << endl;
        multiVertCount = new int*[NUM_CORES];
        Rs = new Random[NUM_CORES];
        for(int i = 0; i < NUM_CORES; i++){
            Rs[i] = Random(unsigned(rand()));
            multiVertCount[i] = new int[vert];
            for(int j = 0; j < vert; j++){
                multiVertCount[i][j] = 0;
            }
        }
        cout << "init done! " << endl;
    }
    ~PPR() {
        for(int i = 0; i < NUM_CORES; i++){
            delete[] multiVertCount[i];
        }
        delete[] multiVertCount;
        delete[] vert_count;
        delete[] value_verts;
        delete[] r;
        delete[] isInQueueBack;
        delete[] rmap_back;
        delete[] Rs;
        delete[] r_hash_arr;
        delete[] r_item_arr;
        delete[] r_max_c;
        delete[] q_vector;
    }

    bool is_file_exist(string fileName)
    {
        ifstream infile(fileName);
        return infile.good();
    }

    //取s点的groundtruth
    vector<int> getRealTopK(int s, int k){
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

    // 取s点用power method计算的准确值 一个向量
    unordered_map<int,double> getRealValue(int s) {
        stringstream ss;
        ss << "ppr-answer/" << target_filename << "/" << s << ".txt";
        string infile = ss.str();
        ifstream real(infile);
        unordered_map<int, double> realValue;
        for(int i=0; i<vert; i++) {
            int tempId;
            double tempSim;
            real >> tempId >> tempSim;
            realValue[tempId] = tempSim;
        }
        real.close();
        return realValue;
    }

    unordered_map<int, double> getRealTopKMap(int s, int k){
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

    void push(int node_count) {
        string inputFile = "dataset/" + target_filename + ".query";
        ifstream node_file(inputFile);
        vector<int> nodes;
        for(int i = 0; i < node_count; i++){
            int temp_node;
            node_file >> temp_node;
            if(g.getOutSize(temp_node) == 0){
                i--;
                cout << "illegal : " << temp_node << endl;
                continue;
            }
            nodes.push_back(temp_node);
        }
        node_file.close();

        double push_threshold = 1e-6;
        int index_threshold = 1000;

        for(int i=0; i<node_count; i++) {
            Forward_push(nodes[i], push_threshold);
            Index_push(nodes[i], push_threshold, index_threshold);
            cout << i << "node count" << endl;
        }
    }

    void Forward_push(int u, double push_threshold) {
        unordered_map<int, double> realMap = getRealValue(u);
        double* resultList = new double[vert];
        for(int i = 0; i < vert; i++)
            resultList[i] = 0;

        double* pi = new double[vert];
        double* res = new double[vert];
        bool* isInQueue = new bool[vert];
        queue<int> r_queue;

        clock_t t0 = clock();
        for(int i=0; i<vert; i++) {
            pi[i] = 0;
            res[i] = 0;
            isInQueue[i] = false;
        }
        res[u] = 1;
        r_queue.push(u);
        isInQueue[u] = true;

        while(r_queue.size()>0) {
            int tempNode = r_queue.front();
            r_queue.pop();
            isInQueue[tempNode] = false;
            int tempDegree = g.getDegree(tempNode);
            double residue = res[tempNode];
            double reserve = q_vector[tempNode]/(q_vector[tempNode]+tempDegree)*residue;
            pi[tempNode] += reserve;
            res[tempNode] = 0;
            residue -= reserve;
            double update = residue/tempDegree;
            for(int i=0; i<tempDegree; i++) {
                int updateNode = g.getNeighbor(tempNode, i);
                res[updateNode] += update;
                if(!isInQueue[updateNode] && res[updateNode] >= push_threshold) {
                    r_queue.push(updateNode);
                    isInQueue[updateNode] = true;
                }
            }
        }
        clock_t t1 = clock();

        double avg_res = 0.;
        double max_res = 0.;
        for(int i=0; i<vert; i++) {
            avg_res += res[i];
            if(res[i] > max_res) max_res=res[i];
        }
        avg_res /= (double)vert;
        // cout << "------------------------ distribution of residual ---------------------------------" << endl;
        // cout << "avg_res: " << avg_res << endl;
        // cout << "max_res: " << max_res << endl;
        // for(int i=1300; i<1400; i++) {
        //     cout << res[i] << "   "  << g.getDegree(i) << endl;
        // }

        for(int i=0; i<vert; i++) {
            if(res[i]>avg_res) {
                int tempNode = i;
                int tempDegree = g.getDegree(tempNode);
                double residue = res[tempNode];
                double reserve = q_vector[tempNode]/(q_vector[tempNode]+tempDegree)*residue;
                pi[tempNode] += reserve;
                res[tempNode] = 0;
                residue -= reserve;
                double update = residue/tempDegree;
                for(int i=0; i<tempDegree; i++) {
                    int updateNode = g.getNeighbor(tempNode, i);
                    res[updateNode] += update;
                }
            }
        }

        avg_res = 0.;
        max_res = 0.;
        for(int i=0; i<vert; i++) {
            avg_res += res[i];
            if(res[i] > max_res) max_res=res[i];
        }
        avg_res /= (double)vert;
        // cout << "------------------------ distribution of residual ---------------------------------" << endl;
        // cout << "avg_res: " << avg_res << endl;
        // cout << "max_res: " << max_res << endl;

        vector<pair<int, double> > vecCluster;
        for(int j = 0; j < vert; j++){
            double rho = pi[j];
            if(rho>0) {
                vecCluster.push_back(pair<int, double>(j, pi[j]));
                // vecCluster.push_back(pair<int, double>(j, pi[j]/g.getDegree(j))); // degree normalized
            }
        }
        sort(vecCluster.begin(), vecCluster.end(), maxScoreCmp);
        // cout << "vecCluster size: " << vecCluster.size() << endl;

        double volS = 0.;
        double cutS = 0.;
        double conductance = 1.0;
        int cutPoint=0;
        int totalVol = 2*g.m;
        unordered_map<int, bool> S;
        int count = 0;
        for(auto& p: vecCluster) {
            // compute conductance
            int v = p.first;
            volS += g.getDegree(v);
            for(int i=0; i<g.getDegree(v); i++) {
                int u = g.getNeighbor(v,i);
                if(S.find(u)==S.end())
                    cutS++;
                else
                    cutS--;
            }
            S[v]=true;

            double cur_conductance = cutS/min(volS, totalVol-volS);
            // cout << cur_conductance << endl;
            if((cur_conductance<conductance) && cur_conductance>0) {
                conductance=cur_conductance;
                cutPoint = count;
            }
            count++;
        }

        avg_clustersize += cutPoint;
        avg_conductance += conductance;
        // cout << "cutPoint: " << cutPoint << endl;
        cout << "conductance: " << conductance << endl;
    
        double l1error = 0.;
        for(int i = 0; i < vert; i++){
            resultList[i] = pi[i];
        }
        double abs_err = 0.;
        for(int i = 0; i<vert; i++){
            double abs_error = abs(realMap[i] - resultList[i]);
            abs_err += abs_error;
            if(abs_error>l1error) l1error = abs_error;
        }
        avg_abs_err += abs_err;
        cout << "abs_error: " << abs_err << endl;
        cout << "l1_error: " << l1error << endl;

        avg_time +=  (t1 - t0) / (double) CLOCKS_PER_SEC ;
        cout << "time used: " << (t1 - t0) / (double) CLOCKS_PER_SEC << endl;
        avg_L1_error += l1error;
        delete[] pi;
        delete[] res;
        delete[] isInQueue;
    }

    void MC_algorithm1(int node_count) {
        string inputFile = "dataset/" + target_filename + ".query";
        ifstream node_file(inputFile);
        vector<int> nodes;
        for(int i = 0; i < node_count; i++){
            int temp_node;
            node_file >> temp_node;
            if(g.getOutSize(temp_node) == 0){
                i--;
                cout << "illegal : " << temp_node << endl;
                continue;
            }
            nodes.push_back(temp_node);
        }
        node_file.close();

        double montecarlo_walk_num = 10000.0;

        for(int i=0; i<node_count; i++) {
            MC_simplewalk(nodes[i], montecarlo_walk_num);
        }
    }

    void MC_algorithm2(int node_count) {
        string inputFile = "dataset/" + target_filename + ".query";
        ifstream node_file(inputFile);
        vector<int> nodes;
        for(int i = 0; i < node_count; i++){
            int temp_node;
            node_file >> temp_node;
            if(g.getOutSize(temp_node) == 0){
                i--;
                cout << "illegal : " << temp_node << endl;
                continue;
            }
            nodes.push_back(temp_node);
        }
        node_file.close();

        double montecarlo_walk_num = 100.0;

        cout << "Monte Carlo loop-erased walk" << endl;

        vector<vector<double> > resultList(node_count, vector<double>(vert, 0.));
        vector<vector<double> > resultList_Compare(node_count, vector<double>(vert, 0.));

        clock_t t0 = clock();
        for(int i=0; i<montecarlo_walk_num; i++) {
            stringstream ss;
            // ss << "index/" << target_filename << "/" << i << ".txt";
            ss << "index/" << target_filename << "/" << i << ".bin";
            string outputFile = ss.str();
            FILE *out = fopen(outputFile.c_str(), "wb");
            // ofstream fout(outputFile);
            vector<int> root = MC_looperased_walk();
            unordered_map<int, double> partition;
            for(int k=0; k<vert; k++) {
                if(partition.find(root[k]) == partition.end()) {
                    partition[root[k]] = 0.;
                }
                partition[root[k]] += q_vector[k];
            }
            // cout << "Partition Size: " << partition.size() << endl;
            for(int k=0; k<node_count; k++) {
                for(int j=0; j<vert; j++) {
                    if(root[nodes[k]] == j) resultList_Compare[k][j] += 1; // estimator 1
                    if(root[nodes[k]] == root[j]) resultList[k][j] += q_vector[j]/partition[root[j]]; // estimator 2
                }
            }
            /*for(int j=0; j<vert; j++) {
                fout << root[j] << "\n";
            }
            fout.close();*/
            int *tree = new int[vert]; memset(tree, 0, sizeof(int)*vert);
            for(int k=0; k<vert; k++) tree[k] = root[k];

            fwrite(tree, sizeof(int), vert, out);
            fclose(out);
            partition.clear();
            delete[] tree;
        }
        for(int k=0; k<node_count; k++) {
            unordered_map<int, double> realMap = getRealValue(nodes[k]);
            double l1error = 0.;
            double l1error_Compare = 0.;
            for(int j=0; j<vert; j++) {
                resultList[k][j] /= montecarlo_walk_num;
                resultList_Compare[k][j] /= montecarlo_walk_num;
                double abs_error = abs(realMap[j] - resultList[k][j]);
                double abs_error_Compare = abs(realMap[j] - resultList_Compare[k][j]);
                avg_abs_err += abs_error;
                avg_abs_err_Compare += abs_error_Compare;
                if(abs_error>l1error) l1error = abs_error;
                if(abs_error_Compare>l1error_Compare) l1error_Compare = abs_error_Compare;
            }
            avg_L1_error += l1error;
            avg_L1_error_Compare += l1error_Compare;
        }
        clock_t t1 = clock();
        avg_time +=  (t1 - t0) / (double) CLOCKS_PER_SEC;
    }

    void MC_simplewalk(int u, double walk_num){
        unordered_map<int, double> realMap = getRealValue(u);
        double sum = 0.;
        for(int i=0; i<vert; i++) {
            sum += realMap[i];
        }
        cout << "Sum of realmap: " << sum << endl;
        double* resultList = new double[vert];
        for(int i = 0; i < vert; i++)
            resultList[i] = 0;
        if(g.getDegree(u) == 0){
            resultList[u] = alpha;
            cout << "error: 0 degree vertice" << endl;
            // return resultList;
        }

        clock_t t0 = clock();
        for(double i = 0; i < walk_num; i++){
            int tempNode = u;
            while(R.drand() > q_vector[tempNode]/(q_vector[tempNode]+g.getDegree(tempNode))){
                int length = g.getDegree(tempNode);
                int r = R.generateRandom() % length;
                tempNode = g.getNeighbor(tempNode, r);
            }
            resultList[tempNode] += 1;
        }
        clock_t t1 = clock();
        // cout << "MonteCarlo time: " << (t1 - t0) / (double) CLOCKS_PER_SEC << endl;
        double l1error = 0.;
        for(int i = 0; i < vert; i++){
            resultList[i] /= (double) walk_num;
        }

        vector<pair<int, double> > vecCluster;
        for(int j = 0; j < vert; j++){
            double rho = resultList[j];
            if(rho>0) {
                vecCluster.push_back(pair<int, double>(j, resultList[j]));
            }
        }
        sort(vecCluster.begin(), vecCluster.end(), maxScoreCmp);
        cout << "vecCluster size: " << vecCluster.size() << endl;

        double volS = 0.;
        double cutS = 0.;
        double conductance = 1.0;
        int cutPoint=0;
        int totalVol = 2*g.m;
        unordered_map<int, bool> S;
        int count = 0;
        for(auto& p: vecCluster) {
            // compute conductance
            int v = p.first;
            volS += g.getDegree(v);
            for(int i=0; i<g.getDegree(v); i++) {
                int u = g.getNeighbor(v,i);
                if(S.find(u)==S.end())
                    cutS++;
                else
                    cutS--;
            }
            S[v]=true;

            double cur_conductance = cutS/min(volS, totalVol-volS);
            // cout << cur_conductance << endl;
            if((cur_conductance<conductance) && cur_conductance>0) {
                conductance=cur_conductance;
                cutPoint = count;
            }
            count++;
        }

        avg_clustersize += cutPoint;
        avg_conductance += conductance;
        cout << "cutPoint: " << cutPoint << endl;
        cout << "conductance: " << conductance << endl;

        for(int i = 0; i<vert; i++){
            double abs_error = abs(realMap[i] - resultList[i]);
            avg_abs_err += abs_error;
            if(abs_error>l1error) l1error = abs_error;
        }
        // cout << "avg_abs_error: " << avg_abs_err << endl;

        avg_time +=  (t1 - t0) / (double) CLOCKS_PER_SEC ;
        avg_L1_error += l1error;
        // return resultList;
    }

    vector<int> MC_looperased_walk(){

        int nroots = 0;
        vector<int> intree(vert, false);
        vector<int> next(vert, -1);
        vector<int> root(vert, -1);
        vector<int> order(vert);

        for(int i=0; i<vert; i++) order[i]=i;

        for(int i=0; i<vert; i++) {
            int u = order[i];
            while(!intree[u]) {
                if(R.drand() < q_vector[u]/(q_vector[u]+g.getDegree(u))) {
                    intree[u] = true;
                    nroots += 1;
                    root[u] = u;
                    next[u] = 0;
                } else {
                    int length = g.getDegree(u);
                    int r = R.generateRandom() % length;
                    next[u] = g.getNeighbor(u, r);
                    u = next[u];
                }
            }
            int r = root[u];

            u = order[i];
            while(!intree[u]) {
                root[u] = r;
                intree[u] = true;
                u = next[u];
            }
        }

        return root;
    }

    vector<int> HK_looperased_walk(){

        int nroots = 0;
        vector<int> intree(vert, false);
        vector<int> next(vert, -1);
        vector<int> root(vert, -1);
        vector<int> order(vert);

        for(int i=0; i<vert; i++) order[i]=i;

        for(int i=0; i<vert; i++) {
            int u = order[i];
            while(!intree[u]) {
                if(R.drand() < q_vector[u]/(q_vector[u]+g.getDegree(u))) {
                    intree[u] = true;
                    nroots += 1;
                    root[u] = u;
                    next[u] = 0;
                } else {
                    int length = g.getDegree(u);
                    int r = R.generateRandom() % length;
                    next[u] = g.getNeighbor(u, r);
                    u = next[u];
                }
            }
            int r = root[u];

            u = order[i];
            while(!intree[u]) {
                root[u] = r;
                intree[u] = true;
                u = next[u];
            }
        }

        return root;
    }

    void t_PowerMethod(vector<int> nodeList, int iterations){
        for(int i = 0; i < nodeList.size(); i++){
            int tempNode = nodeList[i];
            stringstream ss;
            ss << "ppr-answer/" << target_filename << "/" << tempNode << ".txt";
            string outputFile = ss.str();
            // cout << "file: " << outputFile << endl;
            // PowerMethodK(iterations, outputFile, tempNode, 500);
            Abs_PowerMethodK(iterations, outputFile, tempNode, 500);
            // cout << outputFile << "done!"  << endl;        
        }
    }

    void PowerMethodK(int iterations, string outputFile, int u, int k){
        unordered_map<int, double> map_residual;
        map_residual.clear();
        map_residual[u] = 1.0;

        int num_iter=0;
        double* map_ppr = new double[vert];
        for(int i = 0; i < vert; i++){
            map_ppr[i] = 0;
        }
        while( num_iter < iterations ){
            cout << u << ": iter " << num_iter << endl;
            num_iter++;

            vector< pair<int,double> > pairs(map_residual.begin(), map_residual.end());
            map_residual.clear();
            for(auto &p: pairs){
                if(p.second > 0){
                    map_ppr[p.first] += alpha*p.second;
                    int out_deg = g.getOutSize(p.first);

                    double remain_residual = (1-alpha)*p.second;
                    if(out_deg==0){
                        map_residual[u] += remain_residual;
                    }
                    else{
                        double avg_push_residual = remain_residual / out_deg;
                        for(int i = 0; i < g.getOutSize(p.first); i++){
                            int next = g.getOutVert(p.first, i);
                            map_residual[next] += avg_push_residual;
                        }
                    }
                }
            }
        }
        ofstream fout(outputFile);
        vector<pair<int, double> > pprs;
        for(int j = 0; j < vert; j++){
            pprs.push_back(pair<int, double>(j, map_ppr[j]));
        }
        sort(pprs.begin(), pprs.end(), maxScoreCmp);
        for(int j = 0; j < vert; j++){
            if(pprs[j].second >= 0){
                fout << pprs[j].first << " " << pprs[j].second << "\n";
            }
            /*if(j >= 10000 && pprs[j].second < pprs[499].second){
                break;
            }*/
        }
        fout.close();
        delete[] map_ppr;
    }

    void heatkernel(int s, int t) {
        int num = 1000;
        double estimate = 0.;
        for(int i=0; i<num; i++) {
            vector<int> root = HK_looperased_walk();
            if(root[s] == t) estimate += 1.0/num;
        }
    }

    void comparetime() {
        clock_t t0 = clock();
        int u = R.generateRandom() % vert;
        for(int i=0; i<vert; i++) {
            int tempNode = u;
            while(R.drand() > q_vector[tempNode]/(q_vector[tempNode]+g.getDegree(tempNode))){
                int length = g.getDegree(tempNode);
                int r = R.generateRandom() % length;
                tempNode = g.getNeighbor(tempNode, r);
            }    
        }
        clock_t t1 = clock();
        double time1 =  (t1 - t0) / (double) CLOCKS_PER_SEC;
        cout << "Perfrom simple walk n times: " << time1 << endl;

        clock_t t3 = clock();
        vector<int> root = MC_looperased_walk();
        clock_t t4 = clock();
        double time2 =  (t4 - t3) / (double) CLOCKS_PER_SEC;
        cout << "Perfrom loop-erased walk once: " << time2 << endl;
    }

    void Abs_PowerMethodK(int iterations, string outputFile, int u, int k){
        unordered_map<int, double> map_residual;
        map_residual.clear();
        map_residual[u] = 1.0;

        int num_iter=0;
        double* map_ppr = new double[vert];
        for(int i = 0; i < vert; i++){
            map_ppr[i] = 0;
        }
        while( num_iter < iterations ){
            cout << u << ": iter " << num_iter << endl;
            num_iter++;

            vector< pair<int,double> > pairs(map_residual.begin(), map_residual.end());
            map_residual.clear();
            for(auto &p: pairs){
                if(p.second > 0){
                    double absorption_rate = q_vector[p.first]/(q_vector[p.first]+g.getDegree(p.first));
                    map_ppr[p.first] += absorption_rate*p.second;
                    int deg = g.getDegree(p.first);

                    double remain_residual = (1-absorption_rate)*p.second;
                    double avg_push_residual = remain_residual / deg;
                    for(int i = 0; i < g.getDegree(p.first); i++){
                        int next = g.getNeighbor(p.first, i);
                        map_residual[next] += avg_push_residual;
                    }
                }
                }
        }
        ofstream fout(outputFile);
        vector<pair<int, double> > pprs;
        vector<pair<int, double> > vecCluster;
        for(int j = 0; j < vert; j++){
            double rho = map_ppr[j];
            if(rho>0) {
                vecCluster.push_back(pair<int, double>(j, map_ppr[j]));
            }
            pprs.push_back(pair<int, double>(j, map_ppr[j]));
        }
        sort(pprs.begin(), pprs.end(), maxScoreCmp);
        sort(vecCluster.begin(), vecCluster.end(), maxScoreCmp);
        cout << "vecCluster size: " << vecCluster.size() << endl;
        
        double volS = 0.;
        double cutS = 0.;
        double conductance = 1.0;
        int cutPoint=0;
        int totalVol = 2*g.m;
        unordered_map<int, bool> S;
        int count = 0;
        for(auto& p: vecCluster) {
            // compute conductance
            int v = p.first;
            volS += g.getDegree(v);
            for(int i=0; i<g.getDegree(v); i++) {
                int u = g.getNeighbor(v,i);
                if(S.find(u)==S.end())
                    cutS++;
                else
                    cutS--;
            }
            S[v]=true;

            double cur_conductance = cutS/min(volS, totalVol-volS);
            // cout << cur_conductance << endl;
            if((cur_conductance<conductance) && cur_conductance>0) {
                conductance=cur_conductance;
                cutPoint = count;
            }
            count++;
        }

        // cout << "The conductance is " << conductance << endl;
        avg_clustersize += cutPoint;
        avg_conductance += conductance;
        cout << "cutPoint: " << cutPoint << endl;
        cout << "conductance: " << conductance << endl;

        for(int j = 0; j < vert; j++){
            if(pprs[j].second >= 0){
                fout << pprs[j].first << " " << pprs[j].second << "\n";
            }
            /*if(j >= 10000 && pprs[j].second < pprs[499].second){
                break;
            }*/
        }
        fout.close();
        delete[] map_ppr;
    }

    void Index_push(int u, double push_threshold, int index_threshold) {
        unordered_map<int, double> realMap = getRealValue(u);
        double* resultList = new double[vert];
        for(int i = 0; i < vert; i++)
            resultList[i] = 0;

        double* pi = new double[vert];
        double* res = new double[vert];
        bool* isInQueue = new bool[vert];
        queue<int> r_queue;

        clock_t t0 = clock();
        for(int i=0; i<vert; i++) {
            pi[i] = 0;
            res[i] = 0;
            isInQueue[i] = false;
        }
        res[u] = 1;
        r_queue.push(u);
        isInQueue[u] = true;

        while(r_queue.size()>0) {
            int tempNode = r_queue.front();
            r_queue.pop();
            isInQueue[tempNode] = false;
            int tempDegree = g.getDegree(tempNode);
            double residue = res[tempNode];
            double reserve = q_vector[tempNode]/(q_vector[tempNode]+tempDegree)*residue;
            pi[tempNode] += reserve;
            res[tempNode] = 0;
            residue -= reserve;
            double update = residue/tempDegree;
            for(int i=0; i<tempDegree; i++) {
                int updateNode = g.getNeighbor(tempNode, i);
                res[updateNode] += update;
                if(!isInQueue[updateNode] && res[updateNode] >= push_threshold) {
                    r_queue.push(updateNode);
                    isInQueue[updateNode] = true;
                }
            }
        }

        double* error = new double[vert];
        for(int i=0; i<vert; i++) {
            error[i] = 0;
        }
        for(int i=0; i<index_threshold; i++) {
            stringstream ss;
            // ss << "index/" << target_filename << "/" << i << ".txt";
            // cout << i << endl;
            ss << "index/" << target_filename << "/" << i << ".bin";
            string inputFile = ss.str();

            int* root = new int[vert];
            // ifstream index(inputFile);
            FILE* input = fopen(inputFile.c_str(), "rb");
            /*for(int j=0; j<vert; j++) {
                index >> root[j];
            }
            index.close();*/
            fread(root, sizeof(int), vert, input);
            fclose(input);
            /*unordered_map<int, double> partition;
            int* rootS = new int[vert];
            int rootCount = 0;
            for(int k=0; k<vert; k++) {
                if(partition.find(root[k]) == partition.end()) {
                    partition[root[k]] = 0.;
                    rootS[rootCount++] = root[k];
                }
                partition[root[k]] += res[k];
            }
            for(int k=0; k<rootCount; k++) {
                error[rootS[k]] += partition[rootS[k]]/index_threshold;
                // if(root[nodes[k]] == root[j]) error[j] += q_vector[j]/partition[root[j]];
            }*/
            for(int k=0; k<vert; k++) {
                error[k] += res[root[k]]/(double)index_threshold;
            }
            
            delete[] root;
            // delete[] rootS;
        }

        clock_t t1 = clock();

        for(int i = 0; i < vert; i++){
            resultList[i] = pi[i] + error[i];
        }

        vector<pair<int, double> > vecCluster;
        for(int j = 0; j < vert; j++){
            double rho = resultList[j];
            if(rho>0) {
                vecCluster.push_back(pair<int, double>(j, resultList[j]));
            }
        }
        sort(vecCluster.begin(), vecCluster.end(), maxScoreCmp);
        // cout << "vecCluster size: " << vecCluster.size() << endl;

        double volS = 0.;
        double cutS = 0.;
        double conductance = 1.0;
        int cutPoint=0;
        int totalVol = 2*g.m;
        unordered_map<int, bool> S;
        int count = 0;
        for(auto& p: vecCluster) {
            // compute conductance
            int v = p.first;
            volS += g.getDegree(v);
            for(int i=0; i<g.getDegree(v); i++) {
                int u = g.getNeighbor(v,i);
                if(S.find(u)==S.end())
                    cutS++;
                else
                    cutS--;
            }
            S[v]=true;

            double cur_conductance = cutS/min(volS, totalVol-volS);
            // cout << cur_conductance << endl;
            if((cur_conductance<conductance) && cur_conductance>0) {
                conductance=cur_conductance;
                cutPoint = count;
            }
            count++;
        }

        avg_clustersize += cutPoint;
        avg_conductance += conductance;
        // cout << "cutPoint: " << cutPoint << endl;
        // cout << "conductance: " << conductance << endl;
    
        double l1error = 0.;
        for(int i = 0; i<vert; i++){
            double abs_error = abs(realMap[i] - resultList[i]);
            avg_abs_err += abs_error;
            if(abs_error>l1error) l1error = abs_error;
        }
        // cout << "avg_abs_error: " << avg_abs_err << endl;

        avg_time +=  (t1 - t0) / (double) CLOCKS_PER_SEC ;
        avg_L1_error += l1error;
        delete[] pi;
        delete[] res;
        delete[] isInQueue;
        delete[] error;
        delete[] resultList;
    }

    //generate random query node
    void generateQueryNode(int nodeNum, ofstream& fout){
        for(int i = 0; i < nodeNum; i++){
            int tempNode = R.generateRandom() % vert;
            if(g.getOutSize(tempNode) == 0){
                i--;
                continue;   
            }
            fout << tempNode << endl;
        }
    }
};

void ppr_t_PowerMethod(PPR* ppr, vector<int> nodeList, int iterations){
    return ppr->t_PowerMethod(nodeList, iterations);
}

void PPR::PowerMethodMulti(int iterations, int node_count, int num_thread){
    struct timeval t_start,t_end; 
    gettimeofday(&t_start, NULL); 
    long start = ((long)t_start.tv_sec)*1000+(long)t_start.tv_usec/1000; 
    string inputFile = "dataset/" + target_filename + ".query";
    ifstream node_file(inputFile);
    vector<int> nodes;
    for(int i = 0; i < node_count; i++){
        int temp_node;
        node_file >> temp_node;
        if(g.getOutSize(temp_node) == 0){
            i--;
            cout << "illegal : " << temp_node << endl;
            continue;
        }
        nodes.push_back(temp_node);
    }
    node_file.close();
    if(node_count < num_thread){
        num_thread = node_count;
    }
    vector<thread> threads;
    for(int i = 0; i < num_thread-1; i++){
        vector<int> t_nodes;
        for(int j = 0; j < node_count / num_thread; j++){
            t_nodes.push_back(nodes[i * node_count / num_thread + j]);
        }
        threads.push_back(thread(ppr_t_PowerMethod, this, t_nodes, iterations));
    }
    vector<int> t_nodes;
    for(int j = 0; j < node_count / num_thread; j++){
        t_nodes.push_back(nodes[(num_thread-1) * node_count / num_thread + j]);
    }
    t_PowerMethod(t_nodes, iterations);
    for (int i = 0; i < num_thread - 1; i++){
        threads[i].join();
    }
    gettimeofday(&t_end, NULL); 
    long end = ((long)t_end.tv_sec)*1000+(long)t_end.tv_usec/1000; 
    int cost_time = end - start;

    cout << "cost: " << cost_time / (double) 1000 << endl;
}
#endif
