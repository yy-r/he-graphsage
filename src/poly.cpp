#include "poly.h"

using namespace std;
using namespace seal;



// #CMul = 2, #Mul =2 
void poly_eval_deg_3(
    const Ciphertext& x_encrypted,
    const vector<double>& a, //[a0, a1, a2, a3]
    Ciphertext& result,
    const CKKSEncoder& encoder,
    const Evaluator& evaluator,
    const RelinKeys& relin_keys,
    double scale)
{
    if (a.size() < 4) {
        throw invalid_argument("At least four coefficients are required.");
    }

    vector<double> lower_a(a.begin(), a.begin() + 2);
    vector<double> higher_a(a.begin() + 2, a.begin() + 4);
    poly_eval_deg_1(x_encrypted, lower_a, result, encoder, evaluator, scale); // a[0] + a[1]*x
    Ciphertext higher_part;
    poly_eval_deg_1(x_encrypted, higher_a, higher_part, encoder, evaluator, scale); // a[2] + a[3]*x
    Ciphertext x_square;
    evaluator.square(x_encrypted, x_square);
    evaluator.relinearize_inplace(x_square, relin_keys);
    evaluator.rescale_to_next_inplace(x_square);
    x_square.scale() = scale;
    evaluator.multiply_inplace(higher_part, x_square); // (a[2] + a[3]*x)*(x^2)
    evaluator.relinearize_inplace(higher_part, relin_keys);
    evaluator.rescale_to_next_inplace(higher_part);
    higher_part.scale() = scale;
    modswitch_to_match_and_set_scale(evaluator, higher_part, result);
    evaluator.add_inplace(result, higher_part);
}

void poly_eval_deg_1(
    const Ciphertext& x_encrypted,
    const vector<double>& a, //[a0, a1]
    Ciphertext& result,
    const CKKSEncoder& encoder,
    const Evaluator& evaluator,
    double scale)
{
    if (a.size() < 2) {
        throw invalid_argument("At least two coefficients are required.");
    }
    Plaintext plain_a;
    encoder.encode(a[1], x_encrypted.parms_id(), scale, plain_a);
    evaluator.multiply_plain(x_encrypted, plain_a, result);
    evaluator.rescale_to_next_inplace(result);
    result.scale() = scale;
    encoder.encode(a[0], result.parms_id(), scale, plain_a);
    evaluator.add_plain_inplace(result, plain_a);
}
void compute_polynomial_3th_degree(
    const vector<vector<Ciphertext>>& x_encrypted,
    vector<double> coefficients,
    vector<vector<Ciphertext>>& result,
    const CKKSEncoder& encoder,
    const Evaluator& evaluator,
    const RelinKeys& relin_keys,
    double scale) {

        for (int i = 0; i < x_encrypted.size(); i++) {
            for (int j = 0; j < x_encrypted[0].size(); j++) {
                poly_eval_deg_3(x_encrypted[i][j], coefficients, result[i][j], encoder, evaluator, relin_keys, scale);
            }
        }
}