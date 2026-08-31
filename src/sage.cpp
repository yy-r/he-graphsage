#include "sage.h"
#include <cmath>

using namespace std;
using namespace seal;
using namespace Eigen;

vector<vector<vector<Ciphertext>>> row_encode(const vector<vector<MatrixXd>>& mat, const CKKSEncoder& encoder, const Encryptor& encryptor, double scale) {
    vector<vector<vector<Ciphertext>>> re(mat.size(), vector<vector<Ciphertext>>(mat[0].size(), vector<Ciphertext>(mat[0][0].rows())));
    Plaintext plain;
    for (size_t z = 0; z < mat.size(); z++) {
        for (size_t c = 0; c < mat[0].size(); c++) {
            vector<double> temp(mat[z][c].cols(), 0);
            for (int i = 0; i < mat[z][c].rows(); i++) {
                for (int j = 0; j < mat[z][c].cols(); j++) {
                    temp[j] = mat[z][c](i, j);
                }
                encoder.encode(temp, scale, plain);
                encryptor.encrypt(plain, re[z][c][i]);
            }
        }
    }
    return re;
}

vector<vector<Ciphertext>> block_encode(const MatrixXd& mat, const CKKSEncoder& encoder, const Encryptor& encryptor, double scale, int n, int f) {
    int row = mat.rows();
    int col = mat.cols();
    int rg = (row + n - 1) / n;
    int cg = (col + f - 1) / f;
    vector<vector<Ciphertext>> plain(rg, vector<Ciphertext>(cg));
    Plaintext pt;
    int cst = 0;
    int rst = 0;
    for (int i = 0; i < rg; i++) {
        for (int j = 0; j < cg; j++) {
            vector<double> temp(n * f, 0.0);
            for (int c = 0; c < f; c++) {
                cst = f * j + c;
                if (!(cst < col)) continue;
                for (int r = 0; r < n; r++) {
                    rst = i * n + r;
                    if (!(rst < row)) continue;
                    temp[r * f + c] = mat(rst, cst);
                }
            }
            encoder.encode(temp, scale, pt);
            encryptor.encrypt(pt, plain[i][j]);
        }
    }
    return plain;
}

vector<vector<MatrixXd>> block_diag(const MatrixXd& matrix, int n, int f) {
    int row = matrix.rows();
    int col = matrix.cols();
    int rg = (row + n - 1) / n;
    int block_r = row;
    if (rg > 1) block_r = n;
    int cg = (col + n - 1) / n;
    int cst = 0;
    int rst = 0;
    vector<vector<MatrixXd>> plain(rg, vector<MatrixXd>(cg));
    MatrixXd diag_mat = MatrixXd::Zero(block_r, n);
    MatrixXd full_mat = MatrixXd::Zero(block_r, n * f);
    for (int i = 0; i < rg; i++) {
        for (int j = 0; j < cg; j++) {
            for (int r = 0; r < block_r; r++) {
                for (int c = 0; c < n; c++) {
                    int flag = n * r + c;
                    rst = flag % block_r;
                    cst = (flag + (flag / n)) % n;
                    diag_mat(r, c) = matrix(rst + i * n, cst + j * n);
                }
            }
            for (int k = 0; k < n * f; k++) {
                full_mat.col(k) = diag_mat.col(k / f);
            }
            plain[i][j] = full_mat;
        }
    }
    return plain;
}

void block_Rot(vector<vector<MatrixXd>>& fm, int f) {
    for (size_t z = 0; z < fm.size(); z++) {
        for (size_t c = 0; c < fm[0].size(); c++) {
            int n = fm[z][c].rows();
            int cols = fm[z][c].cols();
            MatrixXd rot_mat = MatrixXd::Zero(n, cols);

            auto [sub1, sub2] = decompose_bsgs(n);

            for (int i = 0; i < sub1 && i * sub2 < n; ++i) {
                for (int j = 0; j < sub2; j++) {
                    int row = i * sub2 + j;
                    if (row >= n) break;
                    for (int k = 0; k < cols; k++) {
                        rot_mat(row, (k + f * i * sub2) % cols) = fm[z][c](row, k);
                    }
                }
            }
            fm[z][c] = rot_mat;
        }
    }
}

vector<Plaintext> encode_W_mask(MatrixXd weight, const CKKSEncoder& encoder, double scale, int n, int f) {
    int rows = weight.rows();
    int cols = weight.cols();
    int num_block_rows = rows / f;
    int num_block_cols = cols / f;
    vector<double> diagonal_vec(n * f);
    vector<double> temp_f(f);
    vector< Plaintext> encoded_weight(num_block_cols * f * num_block_rows);
    for (int c = 0; c < num_block_cols; ++c) {
        for (int k = 0; k < f; ++k) {
            for (int r = 0; r < num_block_rows; ++r) {
                for (int i = 0; i < f; ++i) {
                    int row_idx = r * f + i;
                    int col_idx = c * f + (i + k) % f;
                    temp_f[i] = weight(row_idx, col_idx);
                }
                for (int repeat = 0; repeat < n; ++repeat) {
                    copy(temp_f.begin(), temp_f.end(), diagonal_vec.begin() + repeat * f);
                }
                encoder.encode(diagonal_vec, scale, encoded_weight[c * f * num_block_rows + k * num_block_rows + r]);
            }
        }
    }
    return encoded_weight;
}

encode_weight encode_W_split(MatrixXd weight, const CKKSEncoder& encoder, double scale, int n, int f) {
    int rows = weight.rows();
    int cols = weight.cols();
    int num_block_rows = rows / f;
    int num_block_cols = cols / f;
    vector<double> diagonal_vecr(n * f);
    vector<double> temp_fr(f);
    vector<double> diagonal_vecl(n * f);
    vector<double> temp_fl(f);
    vector<Plaintext> encoded_weightr(num_block_cols * f * num_block_rows);
    vector<Plaintext> encoded_weightl(num_block_cols * f * num_block_rows);
    for (int c = 0; c < num_block_cols; ++c) {
        for (int k = 0; k < f; ++k) {
            for (int r = 0; r < num_block_rows; ++r) {
                for (int i = 0; i < f; ++i) {
                    double mask = 0.0;
                    if (i >= (f - k)) mask = 1.0;
                    int row_idx = r * f + i;
                    int col_idx = c * f + (i + k) % f;
                    temp_fl[i] = weight(row_idx, col_idx) * mask;
                    temp_fr[i] = weight(row_idx, col_idx) - temp_fl[i];
                }
                for (int repeat = 0; repeat < n; ++repeat) {
                    copy(temp_fl.begin(), temp_fl.end(), diagonal_vecl.begin() + repeat * f);
                }
                for (int repeat = 0; repeat < n; ++repeat) {
                    copy(temp_fr.begin(), temp_fr.end(), diagonal_vecr.begin() + repeat * f);
                }
                encoder.encode(diagonal_vecl, scale, encoded_weightl[c * f * num_block_rows + k * num_block_rows + r]);
                encoder.encode(diagonal_vecr, scale, encoded_weightr[c * f * num_block_rows + k * num_block_rows + r]);
            }
        }
    }
    return {encoded_weightl, encoded_weightr};
}

vector<vector<Ciphertext>> MaskXW(vector<vector<Ciphertext>> cipher, MatrixXd weight, const Evaluator& evaluator,
    const GaloisKeys& galois_keys, const CKKSEncoder& encoder, const Encryptor& encryptor, double scale, int n, int f) {
    int pmul = 0;
    int rot = 0;
    auto w_row = weight.rows();
    auto w_col = weight.cols();
    int batch = (w_row + f - 1) / f;
    int rnum = (w_col + f - 1) / f;
    MatrixXd padded = MatrixXd::Zero(batch * f, rnum * f);
    copyMat(padded, weight);
    vector<vector<Ciphertext>> encrypted_result(cipher.size(), vector<Ciphertext>(rnum));
    vector<double> onems(n * f, 1.0);
    Plaintext pone;
    vector<Plaintext> pl(f);
    vector<Plaintext> pr(f);
    for (int i = 1; i < f; i++) {
        vector<double> msl(n * f, 0.0);
        vector<double> msr(n * f, 0.0);
        for (int p = 0; p < n * f; p++) {
            if (p % f < i) {
                msl[p] = 1;
            }
            msr[p] = onems[p] - msl[p];
        }
        encoder.encode(msl, scale, pl[i]);
        encoder.encode(msr, scale, pr[i]);
    }

    vector<double> base(f);
    vector<double> w(n * f);
    Plaintext pw;
    Ciphertext ct, ctl, ctr;
    vector<Ciphertext> sum(f);
    for (size_t r = 0; r < cipher.size(); r++) {
        for (int i = 0; i < rnum; i++) {
            for (int j = 0; j < f; j++) {
                for (int b = 0; b < f; b++) {
                    base[b] = padded(b, i * f + (b + j) % f);
                }
                for (int m = 0; m < n; m++) {
                    copy(base.begin(), base.end(), w.begin() + m * f);
                }
                encoder.encode(w, cipher[r][0].parms_id(), scale, pw);
                evaluator.multiply_plain(cipher[r][0], pw, sum[j]);
                pmul++;
                for (int k = 1; k < batch; k++) {
                    for (int b = 0; b < f; b++) {
                        base[b] = padded(b + k * f, i * f + (b + j) % f);
                    }
                    for (int m = 0; m < n; m++) {
                        copy(base.begin(), base.end(), w.begin() + m * f);
                    }
                    encoder.encode(w, cipher[r][0].parms_id(), scale, pw);
                    evaluator.multiply_plain(cipher[r][k], pw, ct);
                    pmul++;
                    evaluator.add_inplace(sum[j], ct);
                }
                evaluator.rescale_to_next_inplace(sum[j]);
                if (j == 0) {
                    encrypted_result[r][i] = sum[j];
                    encoder.encode(onems, encrypted_result[r][i].parms_id(), scale, pone);
                    evaluator.multiply_plain_inplace(encrypted_result[r][i], pone);
                    pmul++;
                } else {
                    evaluator.rotate_vector(sum[j], f - j, galois_keys, ctl);
                    rot++;
                    evaluator.rotate_vector(sum[j], -j, galois_keys, ctr);
                    rot++;
                    evaluator.mod_switch_to_inplace(pl[j], ctl.parms_id());
                    evaluator.multiply_plain_inplace(ctl, pl[j]);
                    pmul++;
                    evaluator.mod_switch_to_inplace(pr[j], ctr.parms_id());
                    evaluator.multiply_plain_inplace(ctr, pr[j]);
                    pmul++;
                    evaluator.add_inplace(ctl, ctr);
                    evaluator.add_inplace(encrypted_result[r][i], ctl);
                }
            }
            evaluator.rescale_to_next_inplace(encrypted_result[r][i]);
        }
    }
    cout << "pmul:" << pmul << endl;
    cout << "rot:" << rot << endl;
    return encrypted_result;
}

vector<vector<Ciphertext>> SplitXW(vector<vector<Ciphertext>> cipher, MatrixXd weight, const Evaluator& evaluator, const GaloisKeys& galois_keys,
    const CKKSEncoder& encoder, const Encryptor& encryptor, double scale, int n, int f) {
    int rot = 0;
    int pmul = 0;
    auto w_row = weight.rows();
    auto w_col = weight.cols();
    int batch = (w_row + f - 1) / f;
    int rnum = (w_col + f - 1) / f;
    int rr = cipher.size();
    MatrixXd padded = MatrixXd::Zero(batch * f, rnum * f);
    copyMat(padded, weight);
    vector<vector<Ciphertext>> encrypted_result(rr, vector<Ciphertext>(rnum));
    vector<double> baser(f);
    vector<double> wr(n * f);
    vector<Ciphertext> sumr(f);
    vector<double> basel(f);
    vector<double> wl(n * f);
    vector<Ciphertext> suml(f);
    Plaintext pw;
    Ciphertext ctr, ctl, ctrro, ctlro;
    for (int r = 0; r < rr; r++) {
        for (int i = 0; i < rnum; i++) {
            for (int j = 0; j < f; j++) {
                for (int b = 0; b < f; b++) {
                    double mask = 0.0;
                    if (b >= (f - j)) mask = 1.0;
                    basel[b] = padded(b, i * f + (b + j) % f) * mask;
                    baser[b] = padded(b, i * f + (b + j) % f) - basel[b];
                }
                for (int m = 0; m < n; m++) {
                    copy(baser.begin(), baser.end(), wr.begin() + m * f);
                    copy(basel.begin(), basel.end(), wl.begin() + m * f);
                }
                encoder.encode(wr, cipher[r][0].parms_id(), scale, pw);
                evaluator.multiply_plain(cipher[r][0], pw, sumr[j]);
                pmul++;
                encoder.encode(wl, cipher[r][0].parms_id(), scale, pw);
                evaluator.multiply_plain(cipher[r][0], pw, suml[j]);
                pmul++;

                for (int k = 1; k < batch; k++) {
                    for (int b = 0; b < f; b++) {
                        double mask = 0.0;
                        if (b >= (f - j)) mask = 1.0;
                        basel[b] = padded(b + k * f, i * f + (b + j) % f) * mask;
                        baser[b] = padded(b + k * f, i * f + (b + j) % f) - basel[b];
                    }
                    for (int m = 0; m < n; m++) {
                        copy(baser.begin(), baser.end(), wr.begin() + m * f);
                        copy(basel.begin(), basel.end(), wl.begin() + m * f);
                    }
                    encoder.encode(wl, cipher[r][k].parms_id(), scale, pw);
                    evaluator.multiply_plain(cipher[r][k], pw, ctl);
                    pmul++;
                    evaluator.add_inplace(suml[j], ctl);
                    encoder.encode(wr, cipher[r][k].parms_id(), scale, pw);
                    evaluator.multiply_plain(cipher[r][k], pw, ctr);
                    pmul++;
                    evaluator.add_inplace(sumr[j], ctr);
                }
                evaluator.rescale_to_next_inplace(suml[j]);
                evaluator.rescale_to_next_inplace(sumr[j]);
                if (j == 0) {
                    encrypted_result[r][i] = sumr[j];
                } else {
                    evaluator.rotate_vector(suml[j], f - j, galois_keys, ctlro);
                    rot++;
                    evaluator.rotate_vector(sumr[j], -j, galois_keys, ctrro);
                    rot++;
                    evaluator.add_inplace(ctlro, ctrro);
                    evaluator.add_inplace(encrypted_result[r][i], ctlro);
                }
            }
        }
    }
    cout << "pmul:" << pmul << endl;
    cout << "rot:" << rot << endl;
    return encrypted_result;
}

vector<vector<Ciphertext>> MaskXW_prew(vector<vector<Ciphertext>> cipher, MatrixXd weight, const Evaluator& evaluator,
    const GaloisKeys& galois_keys, const CKKSEncoder& encoder, const Encryptor& encryptor, double scale, int n, int f) {
    int rot = 0;
    int pmul = 0;
    auto w_row = weight.rows();
    auto w_col = weight.cols();
    int batch = (w_row + f - 1) / f;
    MatrixXd padded = MatrixXd::Zero(batch * f, w_col);
    copyMat(padded, weight);
    auto pr_weight = encode_W_mask(padded, encoder, scale, n, f);
    int rnum = w_col / f;
    vector<vector<Ciphertext>> encrypted_result(cipher.size(), vector<Ciphertext>(rnum));
    vector<double> onems(n * f, 1.0);
    Plaintext pone;
    vector<Plaintext> pl(f);
    vector<Plaintext> pr(f);
    for (int i = 1; i < f; i++) {
        vector<double> msl(n * f, 0.0);
        vector<double> msr(n * f, 0.0);
        for (int p = 0; p < n * f; p++) {
            if (p % f < i) {
                msl[p] = 1;
            }
            msr[p] = onems[p] - msl[p];
        }
        encoder.encode(msl, scale, pl[i]);
        encoder.encode(msr, scale, pr[i]);
    }
    Ciphertext ct, ctl, ctr;
    vector<Ciphertext> sum(f);
    for (size_t r = 0; r < cipher.size(); r++) {
        for (int i = 0; i < rnum; i++) {
            for (int j = 0; j < f; j++) {
                evaluator.multiply_plain(cipher[r][0], pr_weight[i * f * batch + j * batch], sum[j]);
                pmul++;
                for (int k = 1; k < batch; k++) {
                    evaluator.multiply_plain(cipher[r][k], pr_weight[i * f * batch + j * batch + k], ct);
                    pmul++;
                    evaluator.add_inplace(sum[j], ct);
                }
                evaluator.rescale_to_next_inplace(sum[j]);
                if (j == 0) {
                    encrypted_result[r][i] = sum[j];
                    encoder.encode(onems, encrypted_result[r][i].parms_id(), scale, pone);
                    evaluator.multiply_plain_inplace(encrypted_result[r][i], pone);
                    pmul++;
                } else {
                    evaluator.rotate_vector(sum[j], f - j, galois_keys, ctl);
                    rot++;
                    evaluator.rotate_vector(sum[j], -j, galois_keys, ctr);
                    rot++;
                    evaluator.mod_switch_to_inplace(pl[j], ctl.parms_id());
                    evaluator.multiply_plain_inplace(ctl, pl[j]);
                    pmul++;
                    evaluator.mod_switch_to_inplace(pr[j], ctr.parms_id());
                    evaluator.multiply_plain_inplace(ctr, pr[j]);
                    pmul++;
                    evaluator.add_inplace(ctl, ctr);
                    evaluator.add_inplace(encrypted_result[r][i], ctl);
                }
            }
            evaluator.rescale_to_next_inplace(encrypted_result[r][i]);
        }
    }
    cout << "pmul:" << pmul << endl;
    cout << "rot:" << rot << endl;
    return encrypted_result;
}

vector<vector<Ciphertext>> SplitXW_prew(vector<vector<Ciphertext>> cipher, MatrixXd weight, const Evaluator& evaluator,
    const GaloisKeys& galois_keys, const CKKSEncoder& encoder, const Encryptor& encryptor, double scale, int n, int f) {
    int rot = 0;
    int pmul = 0;
    auto w_row = weight.rows();
    auto w_col = weight.cols();
    int batch = (w_row + f - 1) / f;
    MatrixXd padded = MatrixXd::Zero(batch * f, w_col);
    copyMat(padded, weight);
    auto pr_weight = encode_W_split(padded, encoder, scale, n, f);
    int rnum = w_col / f;
    vector<vector<Ciphertext>> encrypted_result(cipher.size(), vector<Ciphertext>(rnum));
    vector<double> baser(f);
    vector<double> wr(n * f);
    vector<Ciphertext> sumr(f);
    vector<double> basel(f);
    vector<double> wl(n * f);
    vector<Ciphertext> suml(f);
    Plaintext pw;
    Ciphertext ctr, ctl, ctrro, ctlro;
    for (size_t r = 0; r < cipher.size(); r++) {
        for (int i = 0; i < rnum; i++) {
            for (int j = 0; j < f; j++) {
                evaluator.multiply_plain(cipher[r][0], pr_weight.ri[i * f * batch + j * batch], sumr[j]);
                pmul++;
                evaluator.multiply_plain(cipher[r][0], pr_weight.le[i * f * batch + j * batch], suml[j]);
                pmul++;
                for (int k = 1; k < batch; k++) {
                    evaluator.multiply_plain(cipher[r][k], pr_weight.le[i * f * batch + j * batch + k], ctl);
                    pmul++;
                    evaluator.add_inplace(suml[j], ctl);
                    evaluator.multiply_plain(cipher[r][k], pr_weight.ri[i * f * batch + j * batch + k], ctr);
                    pmul++;
                    evaluator.add_inplace(sumr[j], ctr);
                }
                evaluator.rescale_to_next_inplace(suml[j]);
                evaluator.rescale_to_next_inplace(sumr[j]);
                if (j == 0) {
                    encrypted_result[r][i] = sumr[j];
                } else {
                    evaluator.rotate_vector(suml[j], f - j, galois_keys, ctlro);
                    rot++;
                    evaluator.rotate_vector(sumr[j], -j, galois_keys, ctrro);
                    rot++;
                    evaluator.add_inplace(ctlro, ctrro);
                    evaluator.add_inplace(encrypted_result[r][i], ctlro);
                }
            }
        }
    }
    cout << "pmul:" << pmul << endl;
    cout << "rot:" << rot << endl;
    return encrypted_result;
}

vector<vector<Ciphertext>> Vecat(const vector<vector<Ciphertext>>& upfeat, const vector<vector<Ciphertext>>& agfeat) {
    vector<vector<Ciphertext>> re(upfeat.size());
    for (size_t i = 0; i < upfeat.size(); i++) {
        re[i].reserve(upfeat[i].size() + agfeat[i].size());
        re[i].insert(re[i].end(), upfeat[i].begin(), upfeat[i].end());
        re[i].insert(re[i].end(), agfeat[i].begin(), agfeat[i].end());
    }
    return re;
}

vector<vector<Ciphertext>> diag_mul_block(vector<vector<vector<Ciphertext>>> diag_cipher, vector<vector<Ciphertext>> col_cipher, const Evaluator& evaluator,
    const GaloisKeys& galois_keys, const CKKSEncoder& encoder, const Encryptor& encryptor, const RelinKeys& relin_keys, double scale, int n, int f) {
    int rot = 0;
    int cmul = 0;
    int slot = n * f;
    int cnum = diag_cipher[0][0].size();
    auto [sub1, sub2] = decompose_bsgs(cnum);
    int rr = diag_cipher.size();
    int rc = col_cipher[0].size();
    int subg = n / cnum;
    vector<vector<Ciphertext>> result(rr, vector<Ciphertext>(rc));
    vector<vector<vector<Ciphertext>>> rot_cipher(col_cipher.size(), vector<vector<Ciphertext>>(rc, vector<Ciphertext>(sub2)));
    for (size_t r = 0; r < col_cipher.size(); r++) {
        for (int i = 0; i < rc; i++) {
            for (int j = 0; j < sub2; j++) {
                evaluator.rotate_vector(col_cipher[r][i], j * f, galois_keys, rot_cipher[r][i][j]);
                rot++;
            }
        }
    }

    vector<Ciphertext> temp(sub1);
    vector<Ciphertext> tempc(col_cipher.size());
    Ciphertext tempx, temps;
    for (int r = 0; r < rr; r++) {
        for (int i = 0; i < rc; i++) {
            for (size_t c = 0; c < col_cipher.size(); c++) {
                for (int j = 0; j < sub1; j++) {
                    modswitch_to_match_and_set_scale(evaluator, rot_cipher[c][i][0], diag_cipher[r][c][0 + sub2 * j]);
                    evaluator.multiply(diag_cipher[r][c][0 + sub2 * j], rot_cipher[c][i][0], tempx);
                    cmul++;
                    for (int z = 1; z < sub2; z++) {
                        modswitch_to_match_and_set_scale(evaluator, rot_cipher[c][i][z], diag_cipher[r][c][z + sub2 * j]);
                        evaluator.multiply(diag_cipher[r][c][z + sub2 * j], rot_cipher[c][i][z], temps);
                        cmul++;
                        evaluator.add_inplace(tempx, temps);
                        if (sub2 * j + z == cnum - 1)
                            break;
                    }
                    evaluator.relinearize_inplace(tempx, relin_keys);
                    evaluator.rotate_vector(tempx, sub2 * j * f, galois_keys, temp[j]);
                    rot++;
                }
                evaluator.add_many(temp, tempc[c]);
            }
            evaluator.add_many(tempc, result[r][i]);
            evaluator.rescale_to_next_inplace(result[r][i]);
            if (subg < 1) subg = 1;
            for (int k = 0; k < (int(log2(subg))); k++) {
                auto rotemp = result[r][i];
                evaluator.rotate_vector_inplace(result[r][i], slot / int(pow(2, k + 1)), galois_keys);
                rot++;
                evaluator.add_inplace(result[r][i], rotemp);
            }
        }
    }
    cout << "cmul:" << cmul << endl;
    cout << "rot:" << rot << endl;
    return result;
}

vector<vector<Ciphertext>> AddBias(vector<vector<Ciphertext>> cipher, vector<double> bias, const Evaluator& evaluator, const CKKSEncoder& encoder,
    double scale, int n, size_t f) {
    vector<vector<Ciphertext>> result(cipher.size(), vector<Ciphertext>(cipher[0].size()));
    for (size_t r = 0; r < cipher.size(); r++) {
        for (size_t z = 0; z < cipher[0].size(); z++) {
            vector<double> vb(n * f, 0);
            vector<double> bi(f, 0);
            Plaintext pb;
            for (size_t i = 0; i < min(f, bias.size() - z * f); i++) {
                bi[i] = bias[i + z * f];
            }
            for (int m = 0; m < n; m++) {
                copy(bi.begin(), bi.end(), vb.begin() + m * f);
            }
            encoder.encode(vb, cipher[r][z].parms_id(), cipher[r][z].scale(), pb);
            evaluator.add_plain(cipher[r][z], pb, result[r][z]);
        }
    }
    return result;
}
