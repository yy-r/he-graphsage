#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include "load.h"
#include "sage.h"
using namespace std;
using namespace seal;
void GraphSAGE(int& res_num) {
    // Set up encryption parameters
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {50, 30, 30, 30, 30, 30, 30, 30, 30, 30, 60 }));
    SEALContext context(parms);
    auto gal_vector = genRot(16, 128);
    KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys galois_keys;
    keygen.create_galois_keys(gal_vector, galois_keys);
    Encryptor encryptor(context, public_key);
    CKKSEncoder encoder(context);
    Evaluator evaluator(context, encoder);
    Decryptor decryptor(context, secret_key);
    double scale = pow(2.0, 30);
    int slot = encoder.slot_count();

    // Load data
    auto dateset = load_dataset("pubmed");
    string prefix = "../model_parameter/pubmed/";
    size_t pad_num = 512;
    int pad_test = 128;
    int sub_n = pad_test;
    int k = 512;
    int l = slot / k;
    if (pad_test > k) sub_n = k;

    auto featmat = vectoMat(dateset.feats);
    auto feat_num = dateset.feats[0].size();
    auto total_num = dateset.feats.size();
    auto class_num = *max_element(dateset.labels.begin(), dateset.labels.end()) + 1;

    auto sage1_weight = read_txt_2d(prefix + "graphsage_sage_layer1_weight.txt", 32, feat_num * 2);
    auto sage2_weight = read_txt_2d(prefix + "graphsage_sage_layer2_weight.txt", 16, 32 * 2);
    auto clas_weight  = read_txt_2d(prefix + "classification_weight.txt", class_num, 16);
    auto clas_bias    = read_txt_1d(prefix + "classification_bias.txt", class_num);
    auto coef1        = read_txt_1d(prefix + "coef1.txt", 4);
    auto coef2        = read_txt_1d(prefix + "coef2.txt", 4);
    // Data preprocess
    auto data_process_t0 = chrono::high_resolution_clock::now();

    auto sample_data  = neighs_sample(dateset.test_index, dateset.adj_map, 8, 8, total_num);
    auto list_mat1    = onehot_Mat(sample_data.test_index, total_num);
    auto list_mat2    = onehot_Mat(sample_data.first_neighs, total_num);
    auto list_mat3    = onehot_Mat(sample_data.second_neighs, total_num);
    auto process_data = Preprocess(list_mat1, list_mat2, list_mat3, featmat, sample_data.premask1, sample_data.premask2);

    auto data_process_t1 = chrono::high_resolution_clock::now();
    cout << "First neighbors size: " << sample_data.first_neighs.size() << endl;

    auto data_process_cost = chrono::duration_cast<chrono::milliseconds>(data_process_t1 - data_process_t0).count();
    std::cout << "Data preprocess cost time: " << data_process_cost << " ms" << endl;

    if (sample_data.first_neighs.size() <= pad_num) {
        auto pad_data = padding(process_data.cat_Firfeats, process_data.normask1, process_data.uplist1_mat1,
                                pad_num - sample_data.first_neighs.size(), pad_test);

        
        auto diagn = block_diag(pad_data.normask1, k, l);
        auto diagu = block_diag(pad_data.uplist1_mat1, k, l);
        block_Rot(diagn, l);
        block_Rot(diagu, l);

        auto encode_t0 = chrono::high_resolution_clock::now();

        auto ciphern = row_encode(diagn, encoder, encryptor, scale);
        auto cipheru = row_encode(diagu, encoder, encryptor, scale);
        auto encry_cat = block_encode(pad_data.cat_Firfeats, encoder, encryptor, scale, k, l);
        auto encode_t1 = chrono::high_resolution_clock::now();
        auto encode_cost = chrono::duration_cast<chrono::milliseconds>(encode_t1 - encode_t0).count();
        std::cout << "encode cost time: " << encode_cost << " ms" << endl;
        
        // --- ciphertext-only timing ---
        auto start_cipher_time = chrono::high_resolution_clock::now();

        auto upfeat1 = MaskXW(encry_cat, sage1_weight.transpose(), evaluator, galois_keys, encoder, encryptor, scale, k, l);

        auto Mask_t = chrono::high_resolution_clock::now();
        auto Mask_cost = chrono::duration_cast<chrono::milliseconds>(Mask_t - start_cipher_time).count();
        std::cout << "XW cost time: " << Mask_cost << " ms" << endl;

        vector<vector<Ciphertext>> feat_relu1(upfeat1.size(), vector<Ciphertext>(upfeat1[0].size()));
        compute_polynomial_3th_degree(upfeat1, coef1, feat_relu1, encoder, evaluator, relin_keys, scale);

        auto aggfeatg0 = diag_mul_block(ciphern, feat_relu1, evaluator, galois_keys, encoder, encryptor, relin_keys, scale, k, l);
        auto upfeatg0 = diag_mul_block(cipheru, feat_relu1, evaluator, galois_keys, encoder, encryptor, relin_keys, scale, k, l);
        auto catVec = Vecat(upfeatg0, aggfeatg0);

        auto XW2_t0 = chrono::high_resolution_clock::now();

        auto sagefeatg = SplitXW(catVec, sage2_weight.transpose(), evaluator, galois_keys, encoder, encryptor, scale, sub_n, l);

        auto XW2_t1 = chrono::high_resolution_clock::now();
        auto XW2_cost = chrono::duration_cast<chrono::milliseconds>(XW2_t1 - XW2_t0).count();
        std::cout << "XW 2 cost time: " << XW2_cost << " ms" << endl;

        vector<vector<Ciphertext>> feat_relu2(sagefeatg.size(), vector<Ciphertext>(sagefeatg[0].size()));
        compute_polynomial_3th_degree(sagefeatg, coef2, feat_relu2, encoder, evaluator, relin_keys, scale);
        auto sagefeat2 = SplitXW(feat_relu2, clas_weight.transpose(), evaluator, galois_keys, encoder, encryptor, scale, sub_n, l);
        sagefeat2 = AddBias(sagefeat2, clas_bias, evaluator, encoder, scale, sub_n, l);

        auto end_time = chrono::high_resolution_clock::now();
        auto cipher_cost = chrono::duration_cast<chrono::milliseconds>(end_time - start_cipher_time).count();
        std::cout << "cipher calculate cost time: " << cipher_cost << " ms" << endl;
        // Decrypt and evaluate
        auto result     = decrypt_vec(sagefeat2, encoder, decryptor, slot);
        auto pr_result  = process_re(result, class_num, k, l, sample_data.test_index.size());
        auto predict    = argmax_axis(pr_result);
        auto true_label = label_select(dateset.labels, dateset.test_index);
        auto acc = acc_calculate(predict, true_label);
        auto auc = auc_calculate_macro(pr_result, true_label, class_num);
        cout << "auc: " << auc << endl;
        cout << "acc: " << acc << endl;

        res_num++;
    }
}
int main() {
    int res_num = 0;
    while(res_num < 10) {
        GraphSAGE(res_num);
    }

}