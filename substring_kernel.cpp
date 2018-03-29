#include<iostream>
#include<cstdlib>
#include<cstdio>
#include<string>
#include<vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

int min(int a, int b){
	if (a < b) return a;
	else return b;
}


std::vector<int> positions(std::string s, char c){
	std::vector<int> v;
	for (unsigned int i = 0; i < s.length(); i++){
		if(s[i] == c){
			v.push_back(i);
		}
	}
	return v;
}



double calcEntry(double lbda, int  word_size, std::string s1, std::string s2){
    int p = word_size;
    int q = s1.length();
    int r = s2.length();
    double B[p+1][q+1][r+1] = {0};
	
    for (int j = 0; j <= q; j++){
        for (int k = 0; k <= r; k++){
            B[0][j][k] = 1;
        }
    }

            
    for (int i = 1; i <= p; i++){
        for (int j = 1; j <= q; j++){
        	for (int k = 1; k <= r; k++){
            	if(i > min(j,k)){
                    B[i][j][k] = 0;
                }
                else{
                    B[i][j][k] = lbda*B[i][j-1][k] + lbda*B[i][j][k-1]- lbda*lbda*B[i][j-1][k-1] + (s1[j-1] == s2[k-1])*B[i-1][j-1][k-1];
                }
            }
        }
    }


	double K[q+1];
	for(int i = 1; i <= q; i++){
	    std::vector<int> idxs = positions(s2, s1[i-1]);
	    double tmp = 0;
	   	for(unsigned int j = 0; j < idxs.size(); j++) {
	   		tmp += B[p-1][i-1][idxs[j]];
		}
		K[i] = K[i-1] + lbda*lbda*tmp;

	}
    return(K[q]);

}


py::array_t<double> getKernel(double lbda, int word_size, py::array chainsA_py, py::array chainsB_py){
	std::vector<std::string> chainsA;
	std::vector<std::string> chainsB;

	for (auto item : chainsA_py){
        //std::cout << "key=" << std::string(py::str(item)) << std::endl;
        chainsA.push_back(std::string(py::str(item)));
	}

	for (auto item : chainsB_py){
        //std::cout << "key=" << std::string(py::str(item)) << std::endl;
        chainsB.push_back(std::string(py::str(item)));
	}		


	int n = chainsA.size();
	int m = chainsB.size();
	double* K = new double[n*m];
	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			K[i*m + j] = calcEntry(lbda, word_size, chainsA[i], chainsB[j]);
		}
	}
	return py::array(n*m, K);
}



PYBIND11_MODULE(substring_kernel, m) {
    m.def("getKernel", &getKernel, "Compute substring kernel");
} 