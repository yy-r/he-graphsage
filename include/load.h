#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <Eigen/Dense>
#include "tool.h"

using namespace Eigen;
using namespace std;

struct Dataset {
    vector<vector<double>> feats;
    vector<int> labels;
    map<int, vector<int>> adj_map;
    vector<int> test_index;
};

vector<int> select_data(string name, int num, int test_num);
Dataset load_dataset(const string& dataset);
MatrixXd read_txt_2d(const string& filename, int d1, int d2);
vector<double> read_txt_1d(const string& filename, int d1);
vector<int> genRot(int f, int test_num);
