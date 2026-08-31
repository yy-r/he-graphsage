#pragma once
#include <vector>
#include <seal/seal.h>

using namespace std;
using namespace seal;

// Inline helpers — small, called in hot loops
static inline void modswitch_to_match_and_set_scale(const Evaluator& evaluator,
    const Ciphertext& ref, Ciphertext& src)
{
    if (src.parms_id() != ref.parms_id())
        evaluator.mod_switch_to_inplace(src, ref.parms_id());
    src.scale() = ref.scale();
}

static inline void align_for_add(SEALContext& context, Evaluator& evaluator, Ciphertext& a, Ciphertext& b, double target_scale)
{
    if (a.parms_id() != b.parms_id()) {
        auto ca = context.get_context_data(a.parms_id());
        auto cb = context.get_context_data(b.parms_id());
        if (ca->chain_index() < cb->chain_index())
            evaluator.mod_switch_to_inplace(b, a.parms_id());
        else if (cb->chain_index() < ca->chain_index())
            evaluator.mod_switch_to_inplace(a, b.parms_id());
    }
    a.scale() = target_scale;
    b.scale() = target_scale;
}

void poly_eval_deg_3(const Ciphertext& x_encrypted, const vector<double>& a, //[a0, a1, a2, a3]
    Ciphertext& result,
    const CKKSEncoder& encoder,
    const Evaluator& evaluator,
    const RelinKeys& relin_keys,
    double scale);
void poly_eval_deg_1(
    const Ciphertext& x_encrypted,
    const vector<double>& a, //[a0, a1]
    Ciphertext& result,
    const CKKSEncoder& encoder,
    const Evaluator& evaluator,
    double scale);
void compute_polynomial_3th_degree(
    const vector<vector<Ciphertext>>& x_encrypted,
    vector<double> coefficients,
    vector<vector<Ciphertext>>& result,
    const CKKSEncoder& encoder,
    const Evaluator& evaluator,
    const RelinKeys& relin_keys,
    double scale);
