#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <seal/seal.h>

using namespace std;
using namespace seal;



pair<int, int> decompose_bsgs(int n);
void printMemoryUsage();
void print_vector(const string& name, const vector<double>& vec);
vector<vector<vector<double>>> decrypt_vec(const vector<vector<Ciphertext>>& result, CKKSEncoder& encoder, Decryptor& decryptor, int slot);
vector<vector<double>> process_re(const vector<vector<vector<double>>>& result, size_t clas, size_t n, size_t f, size_t test);
vector<int> argmax_axis(const vector<vector<double>>& data);
vector<int> label_select(vector<int> labels, vector<int> test);
double acc_calculate(vector<int> predict, vector<int> true_label);
void apply_softmax(vector<vector<double>>& logits);
double calculate_binary_auc(const vector<double>& probs, const vector<int>& binary_labels);
double auc_calculate_macro(vector<vector<double>> probs, const vector<int>& true_labels, int num_classes);
