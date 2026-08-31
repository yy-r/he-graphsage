#include "tool.h"
#include <fstream>
#include <cmath>
#include <stdexcept>

using namespace std;
using namespace seal;


pair<int, int> decompose_bsgs(int n) {
    if (n <= 0) {
        throw invalid_argument("decompose_bsgs: n must be positive, got " + to_string(n));
    }
    int sub2 = static_cast<int>(ceil(sqrt(static_cast<double>(n))));
    int sub1 = (n + sub2 - 1) / sub2;
    return {sub1, sub2};
}

void printMemoryUsage() {
    ifstream status_file("/proc/self/status");
    if (!status_file.is_open()) {
        cerr << "Failed to open /proc/self/status" << endl;
        return;
    }

    string line;
    long vm_rss_kb = 0;

    while (getline(status_file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            line.erase(0, line.find_first_of("0123456789"));
            line.erase(line.find_last_not_of("0123456789") + 1);
            if (!line.empty()) {
                vm_rss_kb = stol(line);
            }
        }
    }
    status_file.close();

    double vm_rss_mb = vm_rss_kb / 1024.0;
    cout.precision(2);
    cout << "Current Physical Memory (RSS): " << fixed << vm_rss_mb << " MB" << endl;
}

void print_vector(const string& name, const vector<double>& vec) {
    cout << name << " (size " << vec.size() << "): ";
    for (double val : vec) {
        cout << val << " ";
    }
    cout << endl;
}

vector<vector<vector<double>>> decrypt_vec(const vector<vector<Ciphertext>>& result, CKKSEncoder& encoder, Decryptor& decryptor, int slot) {
    vector<vector<vector<double>>> predict(result.size(), vector<vector<double>>(result[0].size()));
    Plaintext pla;
    for (size_t i = 0; i < result.size(); i++) {
        for (size_t j = 0; j < result[0].size(); j++) {
            decryptor.decrypt(result[i][j], pla);
            encoder.decode(pla, predict[i][j]);
        }
    }
    return predict;
}

vector<vector<double>> process_re(const vector<vector<vector<double>>>& result, size_t clas, size_t n, size_t f, size_t test) {
    vector<vector<double>> predict(test, vector<double>(clas));
    for (size_t r = 0; r < result.size(); r++) {
        for (size_t c = 0; c < result[0].size(); c++) {
            if (test < r * n) break;
            for (size_t i = 0; i < min(test - r * n, n); i++) {
                for (size_t j = 0; j < min(clas - c * f, f); j++) {
                    predict[i + r * n][j + c * f] = result[r][c][j + f * i];
                }
            }
        }
    }
    return predict;
}

vector<int> argmax_axis(const vector<vector<double>>& data) {
    vector<int> indices;
    indices.reserve(data.size());

    for (const auto& row : data) {
        if (row.empty()) {
            throw runtime_error("Cannot perform argmax on an empty row");
        }
        auto max_iterator = max_element(row.begin(), row.end());
        indices.push_back(distance(row.begin(), max_iterator));
    }
    return indices;
}

vector<int> label_select(vector<int> labels, vector<int> test) {
    vector<int> true_label;
    true_label.reserve(test.size());
    for (size_t i = 0; i < test.size(); i++) {
        true_label.push_back(labels[test[i]]);
    }
    return true_label;
}

double acc_calculate(vector<int> predict, vector<int> true_label) {
    int correct_predictions = 0;
    for (size_t i = 0; i < predict.size(); ++i) {
        if (predict[i] == true_label[i]) {
            correct_predictions++;
        }
    }
    return static_cast<double>(correct_predictions) / predict.size();
}

void apply_softmax(vector<vector<double>>& logits) {
    for (auto& row : logits) {
        if (row.empty()) continue;
        double max_val = *max_element(row.begin(), row.end());
        double sum = 0.0;
        for (double& val : row) {
            val = exp(val - max_val);
            sum += val;
        }
        for (double& val : row) {
            val /= sum;
        }
    }
}

double calculate_binary_auc(const vector<double>& probs, const vector<int>& binary_labels) {
    int n = probs.size();
    int n_pos = 0, n_neg = 0;
    for (int label : binary_labels) {
        if (label == 1) n_pos++;
        else n_neg++;
    }

    if (n_pos == 0 || n_neg == 0) return 0.5;

    struct Sample { double prob; int label; };
    vector<Sample> samples(n);
    for (int i = 0; i < n; ++i) {
        samples[i] = {probs[i], binary_labels[i]};
    }

    sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b) {
        return a.prob < b.prob;
    });

    double rank_sum = 0;
    for (int i = 0; i < n; ++i) {
        if (samples[i].label == 1) {
            rank_sum += (i + 1);
        }
    }

    return (rank_sum - (double)n_pos * (n_pos + 1) / 2.0) / ((double)n_pos * n_neg);
}

double auc_calculate_macro(vector<vector<double>> probs, const vector<int>& true_labels, int num_classes) {
    apply_softmax(probs);

    double total_auc = 0.0;
    for (int c = 0; c < num_classes; ++c) {
        vector<double> class_probs;
        vector<int> binary_labels;
        for (size_t i = 0; i < true_labels.size(); ++i) {
            class_probs.push_back(probs[i][c]);
            binary_labels.push_back(true_labels[i] == c ? 1 : 0);
        }
        total_auc += calculate_binary_auc(class_probs, binary_labels);
    }
    return total_auc / num_classes;
}
