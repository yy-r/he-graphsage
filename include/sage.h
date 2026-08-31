#pragma once
#include <vector>
#include <Eigen/Dense>
#include <seal/seal.h>
#include "poly.h"
#include "tool.h"
#include "process.h"
using namespace std;
using namespace seal;
using namespace Eigen;

struct encode_weight {
    vector<Plaintext> le;
    vector<Plaintext> ri;
};

vector<vector<vector<Ciphertext>>> row_encode(const vector<vector<MatrixXd>>& mat, const CKKSEncoder& encoder,
                                               const Encryptor& encryptor, double scale);
vector<vector<Ciphertext>> block_encode(const MatrixXd& mat, const CKKSEncoder& encoder, const Encryptor& encryptor,
                                        double scale, int n, int f);
vector<vector<MatrixXd>> block_diag(const MatrixXd& matrix, int n, int f);
void block_Rot(vector<vector<MatrixXd>>& fm, int f);

vector<Plaintext> encode_W_mask(MatrixXd weight, const CKKSEncoder& encoder, double scale, int n, int f);
encode_weight encode_W_split(MatrixXd weight, const CKKSEncoder& encoder, double scale, int n, int f);

vector<vector<Ciphertext>> MaskXW(vector<vector<Ciphertext>> cipher, MatrixXd weight, const Evaluator& evaluator,
                                  const GaloisKeys& galois_keys, const CKKSEncoder& encoder, const Encryptor& encryptor,
                                  double scale, int n, int f);
vector<vector<Ciphertext>> SplitXW(vector<vector<Ciphertext>> cipher, MatrixXd weight, const Evaluator& evaluator,
                                   const GaloisKeys& galois_keys, const CKKSEncoder& encoder, const Encryptor& encryptor,
                                   double scale, int n, int f);
vector<vector<Ciphertext>> MaskXW_prew(vector<vector<Ciphertext>> cipher, MatrixXd weight, const Evaluator& evaluator,
                                       const GaloisKeys& galois_keys, const CKKSEncoder& encoder, const Encryptor& encryptor,
                                       double scale, int n, int f);
vector<vector<Ciphertext>> SplitXW_prew(vector<vector<Ciphertext>> cipher, MatrixXd weight, const Evaluator& evaluator,
                                        const GaloisKeys& galois_keys, const CKKSEncoder& encoder, const Encryptor& encryptor,
                                        double scale, int n, int f);
vector<vector<Ciphertext>> Vecat(const vector<vector<Ciphertext>>& upfeat, const vector<vector<Ciphertext>>& agfeat);
vector<vector<Ciphertext>> diag_mul_block(vector<vector<vector<Ciphertext>>> diag_cipher, vector<vector<Ciphertext>> col_cipher,
                                          const Evaluator& evaluator, const GaloisKeys& galois_keys, const CKKSEncoder& encoder,
                                          const Encryptor& encryptor, const RelinKeys& relin_keys, double scale, int n, int f);
vector<vector<Ciphertext>> AddBias(vector<vector<Ciphertext>> cipher, vector<double> bias, const Evaluator& evaluator,
                                   const CKKSEncoder& encoder, double scale, int n, size_t f);
