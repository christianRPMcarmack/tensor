
#include <iostream>
#include "tensor.hpp"

int main() {
	tensor<double> tensor;
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

}