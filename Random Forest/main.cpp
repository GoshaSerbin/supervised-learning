#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <unordered_map>

using namespace std;
using type = double;

#define empty_answer 100
#define eps_zero 1e-10
size_t N = 50;					//size of data
size_t D = 2;				//num of features
type lower = -1, upper = 1;	//boundaries of data
size_t detalization = 160;
size_t x_data_stop_size = 1;
struct tree {
	tree* left;
	tree* right;
	size_t answer;
	size_t depth;
	type threshold;	
	size_t feature;
	tree(size_t depth = 0)
	{
		this->depth = depth;
		feature = 0;
		answer = empty_answer;
		threshold = 10000;
		left = nullptr;
		right = nullptr;
	}
};

double d_rand(double min, double max){
	double f = (double)rand() / RAND_MAX;
	return min + f * (max - min);
}

//stop criterion
bool stop(const vector<vector<type>>& x_data) {
	return (x_data.size() <= x_data_stop_size ? true : false);
}

//returns the most common class in y_data, also changes count
size_t most_common(const vector<size_t>& y_data, size_t& count) {
	unordered_map<size_t, size_t> group;
	for (size_t i = 0; i < y_data.size(); ++i) {
		group[y_data[i]]++;
	}
	size_t max_count = 0, most_common = 0;
	for (auto y : group) {
		if (max_count < y.second) {
			most_common = y.first;
			max_count = y.second;
		}
	}
	count = max_count;
	return most_common;
}

size_t most_common(const vector<size_t>& y_data) {
	size_t temp;
	return most_common(y_data, temp);
}

size_t impurityH(const vector<size_t>& y_data) {
	size_t matches_count;
	most_common(y_data, matches_count);
	return y_data.size() - matches_count;
}

//we maximize this function by choosing best fitting feature and threshold
size_t branch(const vector<vector<type>>& x_data, const vector<size_t>& y_data,\
			size_t feature, type threshold) {
	size_t m = x_data.size();
	vector<size_t> yl_data, yr_data;

	yl_data.reserve(m);
	yr_data.reserve(m);
	size_t max_ind = 0;
	for (size_t i = 1; i < m; ++i) {
		if (x_data[i][feature] > x_data[max_ind][feature]) {
			max_ind = i;
		}
	}
	for (size_t i = 0; i < m; ++i) {

		if (i == max_ind || x_data[i][feature] > threshold)
			yr_data.push_back(y_data[i]);
		else
			yl_data.push_back(y_data[i]);
	}

	if (yl_data.size() == 0) return -5;
	return impurityH(y_data) - impurityH(yl_data) - impurityH(yr_data);
}


void fill_tree(tree* decision_tree, \
				const vector<vector<type>>& x_data,\
				const vector<size_t>& y_data) {
	if (stop(x_data)) {
		decision_tree->answer = most_common(y_data);
		return;
	}
	decision_tree->left = new tree(decision_tree->depth + 1);
	decision_tree->right = new tree(decision_tree->depth + 1);
	size_t max_branch = 0;
	for (size_t t = 0; t < x_data.size(); ++t) {
		for (size_t feature = 0; feature < x_data[0].size(); ++feature) {
			type threshold = x_data[t][feature] + eps_zero;
			size_t val = branch(x_data, y_data, feature, threshold);
			if (val >= max_branch) {
				max_branch = val;
				decision_tree->feature = feature;
				decision_tree->threshold = threshold;
			}
		}
	}
	if (max_branch == 0) {	
		decision_tree->answer = most_common(y_data);
		return;
	};
	size_t m = x_data.size();
	vector<size_t> yl_data, yr_data;
	vector<vector<type>> xl_data, xr_data;
	yl_data.reserve(m);
	yr_data.reserve(m);
	xl_data.reserve(m);
	xr_data.reserve(m);
	size_t max_ind = 0;
	for (size_t i = 1; i < m; ++i) {
		if (x_data[i][decision_tree->feature] > x_data[max_ind][decision_tree->feature]) {
			max_ind = i;
		}
	}
	for (size_t i = 0; i < m; ++i) {
		if (i == max_ind || x_data[i][decision_tree->feature] > decision_tree->threshold) {
			yr_data.push_back(y_data[i]);
			xr_data.push_back(x_data[i]);
		}
		else {
			yl_data.push_back(y_data[i]);
			xl_data.push_back(x_data[i]);
		}
	}
	fill_tree(decision_tree->left, xl_data, yl_data);
	
	fill_tree(decision_tree->right, xr_data, yr_data);
}


void fill_tree_in_forest(tree* decision_tree, \
	const vector<vector<type>>& x_data, \
	const vector<size_t>& y_data) {
	if (stop(x_data)) {
		decision_tree->answer = most_common(y_data);
		return;
	}
	decision_tree->left = new tree(decision_tree->depth + 1);
	decision_tree->right = new tree(decision_tree->depth + 1);
	size_t max_branch = 0;
	size_t feature = rand() % x_data[0].size();
	for (size_t t = 0; t < x_data.size(); ++t) {
		type threshold = x_data[t][feature] + eps_zero;
		size_t val = branch(x_data, y_data, feature, threshold);
		if (val >= max_branch) {
			max_branch = val;
			decision_tree->feature = feature;
			decision_tree->threshold = threshold;
		}
	}
	if (max_branch == 0) {
		decision_tree->answer = most_common(y_data);
		return;
	};
	size_t m = x_data.size();
	vector<size_t> yl_data, yr_data;
	vector<vector<type>> xl_data, xr_data;
	yl_data.reserve(m);
	yr_data.reserve(m);
	xl_data.reserve(m);
	xr_data.reserve(m);
	size_t max_ind = 0;
	for (size_t i = 1; i < m; ++i) {
		if (x_data[i][decision_tree->feature] > x_data[max_ind][decision_tree->feature]) {
			max_ind = i;
		}
	}
	for (size_t i = 0; i < m; ++i) {
		if (i == max_ind || x_data[i][decision_tree->feature] > decision_tree->threshold) {
			yr_data.push_back(y_data[i]);
			xr_data.push_back(x_data[i]);
		}
		else {
			yl_data.push_back(y_data[i]);
			xl_data.push_back(x_data[i]);
		}
	}
	fill_tree_in_forest(decision_tree->left, xl_data, yl_data);

	fill_tree_in_forest(decision_tree->right, xr_data, yr_data);
}



void create_data(vector<vector<type>>& x_data, vector<size_t>& y_data, size_t N, size_t D, type lower, type upper) {

	x_data.resize(N);
	for (auto& x : x_data) {
		x.resize(D);
		for (auto& feature : x)
			feature = d_rand(lower, upper);
	}

	y_data.resize(N);
	for (size_t i = 0; i < y_data.size(); ++i) {
		if (x_data[i][0] + x_data[i][1] < -0.5) {
			y_data[i] = 1;
		}
		else {
			if (x_data[i][0] + x_data[i][1] * 3 < 0.7) {
				y_data[i] = 2;
			}
			else
				y_data[i] = 3;
		}

		//if (x_data[i][0] < 0) {
		//	y_data[i] = 1;
		//}
		//else {
		//	if (x_data[i][0] < 0.5) {
		//		y_data[i] = 2;
		//	}
		//	else {
		//		y_data[i] = 3;
		//	}
		//}
	}
}

void write_data(const vector<vector<type>>& x_data, const vector<size_t>& y_data) {
	ofstream out("x_data.txt");
	for (auto const& x : x_data) {
		out << x[0] << " " << x[1] << endl;
	}
	out.close();
	out.open("y_data.txt");
	for (auto const& y : y_data) {
		out << y << endl;
	}
	out.close();
}

void make_predictions(tree* const decision_tree, type lower, type upper) {
	type step = (upper - lower) / (detalization - 1);
	tree* p;

	ofstream out("predictions.txt");
	out << lower << " " << upper << " " << detalization << endl;
	for (type x = lower; x <= upper + step / 2; x += step) {
		for (type y = lower; y <= upper + step / 2; y += step) {
			p = decision_tree;
			while (p->answer == empty_answer) {
				type feature = (p->feature == 0 ? x : y);
				if (feature < p->threshold) {
					p = p->left;
				}
				else {
					p = p->right;
				}
			}
			out << p->answer << " ";
		}
		out << endl;
	}
	out.close();
}

void make_predictions(vector<tree>& decision_trees, type lower, type upper) {
	type step = (upper - lower) / (detalization - 1);
	tree* p;
	ofstream out("predictions.txt");
	out << lower << " " << upper << " " << detalization << endl;
	for (type x = lower; x <= upper + step / 2; x += step) {
		for (type y = lower; y <= upper + step / 2; y += step) {
			vector<size_t> votes;
			for (auto& decision_tree : decision_trees) {
				p = &decision_tree;
				while (p->answer == empty_answer) {
					type feature = (p->feature == 0 ? x : y);
					if (feature < p->threshold) {
						p = p->left;
					}
					else {
						p = p->right;
					}
				}
				votes.push_back(p->answer);
			}
			
			out << most_common(votes) << " ";
		}
		out << endl;
	}
	out.close();
}

void print(tree* decision_tree) {
	if (decision_tree->left != nullptr)
		print(decision_tree->left);
	if (decision_tree->right != nullptr)
		print(decision_tree->right);
	cout << "node: depth=" << decision_tree->depth <<\
		"\tfeature=" << decision_tree->feature <<\
		"\tthreshold=" << decision_tree->threshold<<\
		"\t\tanswer=" << decision_tree->answer << endl;
}
int main() {
	vector<vector<type>> x_data;
	vector<size_t> y_data;
	create_data(x_data, y_data, N, D, lower, upper);
	write_data(x_data, y_data);

	size_t k = 200; //number of trees
	vector<tree> decision_trees;
	decision_trees.resize(k);

	vector<vector<type>> x_sample;
	vector<size_t> y_sample;
	x_sample.resize(N);
	y_sample.resize(N);

	for (auto& x : x_sample) {
		x.resize(D);
	}

	for (int num = 0; num < decision_trees.size(); ++num) {
		for (size_t i = 0; i < N; ++i) {
			size_t seed = rand();
			x_sample[i] = x_data[seed % N];
			y_sample[i] = y_data[seed % N];
		}
		fill_tree_in_forest(&decision_trees[num], x_sample, y_sample);
	}

	make_predictions(decision_trees, lower, upper);

	return 0;
}