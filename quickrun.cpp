
#include <iostream>


#define __DEBUG__TENSOR__
#include "tensor.hpp"

int main() {

	cc::tensor<double> tensor;
	tensor.resize(6, 5, 4);
	auto cnt = 0;
	for (unsigned i = 0; i < tensor.shape()[0]; i++) {
		for (unsigned j = 0; j < tensor.shape()[1]; j++) {
			for (unsigned k = 0; k < tensor.shape()[2]; k++) {
				tensor[i][j][k] = cnt;
				cnt++;
			}
		}
	}

	tensor.resize(3, 3, 3, 3);
	cnt = 0;
	for (unsigned i = 0; i < tensor.shape()[0]; i++) {
		for (unsigned j = 0; j < tensor.shape()[1]; j++) {
			for (unsigned k = 0; k < tensor.shape()[2]; k++) {
				for (unsigned l = 0; l < tensor.shape()[3]; l++) {
					tensor[i][j][k][l] = cnt;
					cnt++;
				}
			}
		}
	}

	cc::tensor<double> tensor2(3, 3, 3, 3);
	cnt = 0;
	for (unsigned i = 0; i < tensor.shape()[0]; i++) {
		for (unsigned j = 0; j < tensor.shape()[1]; j++) {
			for (unsigned k = 0; k < tensor.shape()[2]; k++) {
				for (unsigned l = 0; l < tensor.shape()[3]; l++) {
					tensor2[i][j][k][l] = cnt;
					cnt++;
				}
			}
		}
	}

	tensor += tensor2;

	tensor *= tensor2;

	tensor /= tensor2;

	tensor -= tensor2;
	 
	tensor = tensor2;

	auto t3 = tensor + tensor2;
	auto t4 = tensor - tensor2;
	auto t5 = tensor * tensor2;
	auto t6 = tensor / tensor2;

	for (auto x : t3) {
		std::cout << x << std::endl;
	}
	for (auto x : t4) {
		std::cout << x << std::endl;
	}
	for (auto x : t5) {
		std::cout << x << std::endl;
	}
	for (auto x : t6) {
		std::cout << x << std::endl;
	}
	return 0;
}