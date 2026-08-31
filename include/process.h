#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

struct PreprocessResult {
    MatrixXd cat_Firfeats;
    MatrixXd normask1;
    MatrixXd uplist1_mat1;
};

struct SamplingResult {
    vector<int> test_index;
    vector<int> first_neighs;
    vector<int> second_neighs;
    MatrixXd premask1;
    MatrixXd premask2;
};

struct PaddingResult {
    MatrixXd cat_Firfeats;
    MatrixXd normask1;
    MatrixXd uplist1_mat1;
};


void copyMat(MatrixXd& mat, const MatrixXd& matcopy);
MatrixXd vectoMat(vector<vector<double>> vec);
MatrixXd Concat(const MatrixXd& A, const MatrixXd& B);
MatrixXd onehot_Mat(const vector<int>& nodes, size_t total_num);
MatrixXd rowNormal(const MatrixXd& mask);
PreprocessResult Preprocess(const MatrixXd& list_mat1, const MatrixXd& list_mat2, const MatrixXd& list_mat3,
                            const MatrixXd& feats, const MatrixXd& premask1, const MatrixXd& premask2);
vector<vector<int>> sampling(const vector<vector<int>>& neighs, size_t sample_num);
vector<int> merge(const vector<vector<int>>& neighs, const vector<int>& nodes);
MatrixXd halfMask(const vector<int>& nodes, const vector<vector<int>>& nodeslist, int total_num);
SamplingResult neighs_sample(vector<int> test_index, const map<int, vector<int>>& adj_map,
                             size_t s1, size_t s2, int total_num);
void print_matrix(const string& name, const vector<vector<double>>& matrix);
PaddingResult padding(const MatrixXd& row_pad, const MatrixXd& col_pad1, const MatrixXd& col_pad2,
                      int pad_num, int pad_test);
