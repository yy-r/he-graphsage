#include "load.h"
#include <fstream>
#include <sstream>
#include <random>
#include <stdexcept>
#include <set>

using namespace std;
using namespace Eigen;

std::vector<int> select_data(std::string name, int num, int test_num) {
    std::ifstream pool_file(name);
    std::vector<int> all_indices;
    int val;
    while (pool_file >> val) {
        all_indices.push_back(val);
    }
    pool_file.close();

    std::set<int> used_set;
    std::string used_file_name = "../used_indices/cora.txt";
    std::ifstream used_file_in(used_file_name);
    int used_val;
    if (used_file_in.is_open()) {
        while (used_file_in >> used_val) {
            used_set.insert(used_val);
        }
        used_file_in.close();
    }

    std::vector<int> available_indices;
    for (int idx : all_indices) {
        if (used_set.find(idx) == used_set.end()) {
            available_indices.push_back(idx);
        }
    }

    if (available_indices.size() < (size_t)test_num) {
        std::cout << "Available nodes exhausted. Resetting used_indices.txt..." << std::endl;
        available_indices = all_indices;
        used_set.clear();

        std::ofstream clear_file(used_file_name, std::ios::trunc);
        clear_file.close();
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(available_indices.begin(), available_indices.end(), g);

    std::vector<int> selected_indices(available_indices.begin(), available_indices.begin() + test_num);

    std::ofstream used_file_out(used_file_name, std::ios::app);
    for (int idx : selected_indices) {
        used_file_out << idx << "\n";
    }
    used_file_out.close();

    return selected_indices;
}
Dataset load_dataset(const string& dataset) {
    if (dataset == "cora") {
        string cora_content = "../cora/cora.content";
        string cora_cite = "../cora/cora.cites";

        vector<vector<double>> feats;
        vector<int> labels;
        map<string, int> node_map;
        map<string, int> label_map;

        ifstream content_file(cora_content);
        if (!content_file.is_open()) {
            throw runtime_error("Could not open file: " + cora_content);
        }

        string line;
        int node_counter = 0;
        while (getline(content_file, line)) {
            stringstream ss(line);
            vector<string> info;
            string item;
            while (ss >> item) {
                info.push_back(item);
            }

            vector<double> current_feats;
            for (size_t i = 1; i < info.size() - 1; ++i) {
                current_feats.push_back(stof(info[i]));
            }
            feats.push_back(current_feats);
            node_map[info[0]] = node_counter++;
            if (label_map.find(info.back()) == label_map.end()) {
                label_map[info.back()] = label_map.size();
            }
            labels.push_back(label_map[info.back()]);
        }
        content_file.close();

        map<int, vector<int>> adj_map;
        ifstream cite_file(cora_cite);
        if (!cite_file.is_open()) {
            throw runtime_error("Could not open file: " + cora_cite);
        }

        while (getline(cite_file, line)) {
            stringstream ss(line);
            string id1_str, id2_str;
            ss >> id1_str >> id2_str;

            if (node_map.count(id1_str) && node_map.count(id2_str)) {
                int cite1 = node_map[id1_str];
                int cite2 = node_map[id2_str];
                adj_map[cite1].push_back(cite2);
                adj_map[cite2].push_back(cite1);
            }
        }
        cite_file.close();

        string cora_test = "../cora/cora_test.txt";
        vector<int> test_index = select_data(cora_test, feats.size(), 128);
        sort(test_index.begin(), test_index.end());

        return {feats, labels, adj_map, test_index};
    }
    else if (dataset == "citeseer") {
        string citeseer_content = "../citeseer/citeseer.content";
        string citeseer_cite = "../citeseer/citeseer.cites";

        vector<vector<double>> feats;
        vector<int> labels;
        map<string, int> node_map;
        map<string, int> label_map;

        ifstream content_file(citeseer_content);
        if (!content_file.is_open()) {
            throw runtime_error("Could not open file: " + citeseer_content);
        }

        string line;
        int node_counter = 0;
        while (getline(content_file, line)) {
            stringstream ss(line);
            vector<string> info;
            string item;
            while (ss >> item) {
                info.push_back(item);
            }
            vector<double> current_feats;
            for (size_t i = 1; i < info.size() - 1; ++i) {
                current_feats.push_back(stof(info[i]));
            }
            feats.push_back(current_feats);
            node_map[info[0]] = node_counter++;
            if (label_map.find(info.back()) == label_map.end()) {
                label_map[info.back()] = label_map.size();
            }
            labels.push_back(label_map[info.back()]);
        }
        content_file.close();

        map<int, vector<int>> adj_map;
        ifstream cite_file(citeseer_cite);
        if (!cite_file.is_open()) {
            throw runtime_error("Could not open file: " + citeseer_cite);
        }

        while (getline(cite_file, line)) {
            stringstream ss(line);
            string id1_str, id2_str;
            ss >> id1_str >> id2_str;

            if (node_map.count(id1_str) && node_map.count(id2_str)) {
                int cite1 = node_map[id1_str];
                int cite2 = node_map[id2_str];
                adj_map[cite1].push_back(cite2);
                adj_map[cite2].push_back(cite1);
            }
        }
        cite_file.close();
        string citeseer_test = "../citeseer/citeseer_test.txt";
        vector<int> test_index = select_data(citeseer_test, feats.size(), 128);
        sort(test_index.begin(), test_index.end());

        return {feats, labels, adj_map, test_index};
    }
    else if (dataset == "pubmed") {
        string pubmed_content = "../pubmed/Pubmed-Diabetes.NODE.paper.tab";
        string pubmed_cite = "../pubmed/Pubmed-Diabetes.DIRECTED.cites.tab";

        vector<vector<double>> feats;
        vector<int> labels;
        map<string, int> node_map;
        map<string, int> feat_map;

        ifstream content_file(pubmed_content);
        if (!content_file.is_open()) {
            throw runtime_error("Could not open file: " + pubmed_content);
        }

        string line;
        getline(content_file, line);
        getline(content_file, line);

        stringstream header_ss(line);
        string feat_definition;
        int feat_counter = 0;
        getline(header_ss, feat_definition, '\t');

        while (getline(header_ss, feat_definition, '\t')) {
            size_t first_colon = feat_definition.find(':');
            size_t second_colon = feat_definition.find(':', first_colon + 1);
            if (first_colon != string::npos && second_colon != string::npos) {
                string feat_name = feat_definition.substr(first_colon + 1, second_colon - first_colon - 1);
                feat_map[feat_name] = feat_counter++;
            }
        }

        int node_counter = 0;
        while (getline(content_file, line)) {
            stringstream ss(line);
            vector<string> info;
            string item;
            while (getline(ss, item, '\t')) {
                info.push_back(item);
            }
            if (info.empty() || info[0].empty()) continue;

            node_map[info[0]] = node_counter++;
            size_t eq_pos = info[1].find('=');
            labels.push_back(stoi(info[1].substr(eq_pos + 1)) - 1);

            vector<double> current_feats(feat_map.size(), 0.0);
            for (size_t i = 2; i < info.size() - 1; ++i) {
                eq_pos = info[i].find('=');
                if (eq_pos != string::npos) {
                    string feat_name = info[i].substr(0, eq_pos);
                    double feat_value = stod(info[i].substr(eq_pos + 1));
                    if (feat_map.count(feat_name)) {
                        current_feats[feat_map.at(feat_name)] = feat_value;
                    }
                }
            }
            feats.push_back(current_feats);
        }
        content_file.close();

        map<int, vector<int>> adj_map;
        ifstream cite_file(pubmed_cite);
        if (!cite_file.is_open()) {
            throw runtime_error("Could not open file: " + pubmed_cite);
        }

        getline(cite_file, line);
        getline(cite_file, line);

        while (getline(cite_file, line)) {
            stringstream ss(line);
            vector<string> info;
            string item;
            while (getline(ss, item, '\t')) {
                info.push_back(item);
            }
            if (info.size() < 2) continue;

            string& paper1_info = info[1];
            size_t colon_pos1 = paper1_info.find(':');
            string id1_str = paper1_info.substr(colon_pos1 + 1);

            string& paper2_info = info.back();
            size_t colon_pos2 = paper2_info.find(':');
            string id2_str = paper2_info.substr(colon_pos2 + 1);

            if (node_map.count(id1_str) && node_map.count(id2_str)) {
                int cite1 = node_map.at(id1_str);
                int cite2 = node_map.at(id2_str);
                adj_map[cite1].push_back(cite2);
                adj_map[cite2].push_back(cite1);
            }
        }
        cite_file.close();

        string pubmed_test = "../pubmed/pubmed_test.txt";
        vector<int> test_index = select_data(pubmed_test, feats.size(), 128);
        sort(test_index.begin(), test_index.end());

        return {feats, labels, adj_map, test_index};
    }
    else {
        throw runtime_error("Dataset not supported");
    }
}

MatrixXd read_txt_2d(const string& filename, int d1, int d2) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "can't open file: " << filename << endl;
        exit(1);
    }

    MatrixXd data(d1, d2);
    string line;
    int i = 0;
    while (getline(infile, line) && i < d1) {
        stringstream ss(line);
        for (int j = 0; j < d2; j++) {
            ss >> data(i, j);
        }
        i++;
    }
    infile.close();
    return data;
}

vector<double> read_txt_1d(const string& filename, int d1) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "can't open file: " << filename << endl;
        exit(1);
    }

    vector<double> data(d1);
    for (int i = 0; i < d1; i++) {
        infile >> data[i];
    }
    infile.close();
    return data;
}

vector<int> genRot(int f, int test_num) {
    if (f <= 0) {
        throw invalid_argument("genRot: f must be positive, got " + to_string(f));
    }

    auto [sub1, sub2] = decompose_bsgs(test_num);

    const size_t total_capacity = static_cast<size_t>(2 * f - 1) + sub2 + sub1 + 2;
    vector<int> vs;
    vs.reserve(total_capacity);

    for (int i = 0; i < 2 * f - 1; i++) {
        vs.push_back(-(f - 1) + i);
    }
    for (int i = 0; i < sub2; i++) {
        vs.push_back(i * f);
    }
    for (int i = 0; i < sub1; i++) {
        vs.push_back(i * f * sub2);
    }
    vs.push_back(2048);
    vs.push_back(4096);

    sort(vs.begin(), vs.end());
    vs.erase(unique(vs.begin(), vs.end()), vs.end());

    return vs;
}
