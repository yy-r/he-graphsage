#include "process.h"
#include <random>
#include <stdexcept>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


void copyMat(MatrixXd& mat, const MatrixXd& matcopy) {
    if (mat.rows() >= matcopy.rows() && mat.cols() >= matcopy.cols()) {
        mat.block(0, 0, matcopy.rows(), matcopy.cols()) = matcopy;
    }
}

MatrixXd vectoMat(vector<vector<double>> vec) {
    size_t row = vec.size();
    size_t col = vec[0].size();
    MatrixXd mat(row, col);
    for (size_t i = 0; i < vec.size(); i++) {
        for (size_t j = 0; j < vec[0].size(); j++) {
            mat(i, j) = vec[i][j];
        }
    }
    return mat;
}

MatrixXd Concat(const MatrixXd& A, const MatrixXd& B) {
    if (A.rows() != B.rows()) {
        throw invalid_argument("Matrices must have the same number of rows for horizontal concatenation.");
    }

    size_t rows = A.rows();
    size_t A_cols = A.cols();
    size_t B_cols = B.cols();

    MatrixXd result = MatrixXd::Zero(rows, A_cols + B_cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < A_cols; ++j) {
            result(i, j) = A(i, j);
        }
        for (size_t j = 0; j < B_cols; ++j) {
            result(i, A_cols + j) = B(i, j);
        }
    }
    return result;
}

MatrixXd onehot_Mat(const vector<int>& nodes, size_t total_num) {
    MatrixXd listMatrix = MatrixXd::Zero(nodes.size(), total_num);
    for (size_t i = 0; i < nodes.size(); ++i) {
        size_t row = i;
        size_t col = nodes[i];
        if (col >= 0 && col < total_num) {
            listMatrix(row, col) = 1.0;
        }
    }
    return listMatrix;
}

MatrixXd rowNormal(const MatrixXd& mask) {
    MatrixXd nor_mask = mask;
    for (int i = 0; i < nor_mask.rows(); i++) {
        double sum = 0.0;
        for (int j = 0; j < nor_mask.cols(); j++) {
            sum = sum + nor_mask(i, j);
        }
        if (sum > 1e-6) {
            for (int k = 0; k < nor_mask.cols(); k++) {
                nor_mask(i, k) /= sum;
            }
        }
    }
    return nor_mask;
}

PreprocessResult Preprocess(
    const MatrixXd& list_mat1,
    const MatrixXd& list_mat2,
    const MatrixXd& list_mat3,
    const MatrixXd& feats,
    const MatrixXd& premask1,
    const MatrixXd& premask2)
{
    MatrixXd Fir_feats = list_mat2 * feats;
    MatrixXd Sec_feats = list_mat3 * feats;
    MatrixXd transposed_list_mat3 = list_mat3.transpose();
    MatrixXd mask2 = premask2 * transposed_list_mat3;
    MatrixXd normask2 = rowNormal(mask2);
    MatrixXd agg_Firfeats = normask2 * Sec_feats;
    MatrixXd cat_Firfeats = Concat(Fir_feats, agg_Firfeats);
    MatrixXd transposed_list_mat2 = list_mat2.transpose();
    MatrixXd mask1 = premask1 * transposed_list_mat2;
    MatrixXd normask1 = rowNormal(mask1);
    MatrixXd uplist1_mat1 = list_mat1 * transposed_list_mat2;

    return {cat_Firfeats, normask1, uplist1_mat1};
}

vector<vector<int>> sampling(const vector<vector<int>>& neighs, size_t sample_num) {
    random_device rd;
    mt19937 g(rd());
    vector<vector<int>> sample_neighs;
    sample_neighs.reserve(neighs.size());

    for (const auto& sub : neighs) {
        if (sub.size() <= sample_num) {
            sample_neighs.push_back(sub);
        } else {
            vector<int> sampled_list = sub;
            shuffle(sampled_list.begin(), sampled_list.end(), g);
            sampled_list.resize(sample_num);
            sample_neighs.push_back(sampled_list);
        }
    }
    return sample_neighs;
}

vector<int> merge(const vector<vector<int>>& neighs, const vector<int>& nodes) {
    vector<int> merge_neighs;
    size_t total_size = nodes.size();
    for (const auto& sub : neighs) {
        total_size += sub.size();
    }
    merge_neighs.reserve(total_size);

    for (const auto& sub : neighs) {
        merge_neighs.insert(merge_neighs.end(), sub.begin(), sub.end());
    }
    merge_neighs.insert(merge_neighs.end(), nodes.begin(), nodes.end());

    sort(merge_neighs.begin(), merge_neighs.end());
    merge_neighs.erase(unique(merge_neighs.begin(), merge_neighs.end()), merge_neighs.end());

    return merge_neighs;
}

MatrixXd halfMask(
    const vector<int>& nodes,
    const vector<vector<int>>& nodeslist,
    int total_num)
{
    if (nodes.size() != nodeslist.size()) {
        throw runtime_error("Size of 'nodes' and 'nodeslist' must be the same.");
    }

    MatrixXd mask = MatrixXd::Zero(nodes.size(), total_num);
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (int col : nodeslist[i]) {
            if (col < total_num) {
                mask(i, col) = 1.0;
            }
        }
    }
    return mask;
}

SamplingResult neighs_sample(
    vector<int> test_index,
    const map<int, vector<int>>& adj_map,
    size_t s1,
    size_t s2,
    int total_num)
{
    vector<vector<int>> first_neighs_list;
    first_neighs_list.reserve(test_index.size());
    for (int node : test_index) {
        first_neighs_list.push_back(adj_map.at(node));
    }

    auto sampled_first_neighs = sampling(first_neighs_list, s1);
    auto premask1 = halfMask(test_index, sampled_first_neighs, total_num);
    auto first_neighs = merge(sampled_first_neighs, test_index);

    vector<vector<int>> second_neighs_list;
    second_neighs_list.reserve(first_neighs.size());
    for (int node : first_neighs) {
        second_neighs_list.push_back(adj_map.at(node));
    }

    auto sampled_second_neighs = sampling(second_neighs_list, s2);
    auto premask2 = halfMask(first_neighs, sampled_second_neighs, total_num);
    auto second_neighs = merge(sampled_second_neighs, first_neighs);

    return {test_index, first_neighs, second_neighs, premask1, premask2};
}

void print_matrix(const string& name, const vector<vector<double>>& matrix) {
    cout << name << " (shape " << matrix.size() << "x" << (matrix.empty() ? 0 : matrix[0].size()) << "):" << endl;
    for (const auto& row : matrix) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

PaddingResult padding(
    const MatrixXd& row_pad,
    const MatrixXd& col_pad1,
    const MatrixXd& col_pad2,
    int pad_num, int pad_test)
{
    MatrixXd padded_row = MatrixXd::Zero(row_pad.rows() + pad_num, row_pad.cols());
    copyMat(padded_row, row_pad);

    MatrixXd padded_col1 = MatrixXd::Zero(pad_test, col_pad1.cols() + pad_num);
    copyMat(padded_col1, col_pad1);

    MatrixXd padded_col2 = MatrixXd::Zero(pad_test, col_pad2.cols() + pad_num);
    copyMat(padded_col2, col_pad2);

    return {padded_row, padded_col1, padded_col2};
}
