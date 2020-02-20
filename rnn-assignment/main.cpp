/*
Xue Hang
19M14383
November 2019
ICT.H416 Assignment
Note: Due to my limited programming skills, this code is not well optimised. Please avoid using the
Debug Mode of Visual Studio, as it will take a very long time. Please use the Release Mode instead
and it may take approximately 1 min to run through all 5 tasks. This programme generates .csv data 
files for all 5 tasks, which are used for data analysis.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <cstdlib>
#include <string>
#include <cmath>
#include <bitset>
#include <fstream>

using namespace std;

//Function of random binary sequence generator
vector<int> randBinGen(int seqLeng) {
	vector<int> result;
	int n = 0;
	while (n < seqLeng) {
		srand(time(nullptr));
		result.push_back(rand() % 2);
		n++;
	}
	return result;
}

//Function of random binary generator using sigmoid function for probability
int binDistBinGen(int state, int gain) {
	int result;
	srand(time(nullptr));
	if (rand() <= (RAND_MAX / (1 + exp (- gain * state)))) {
		result = 1;
	}
	else {
		result = 0;
	}
	return result;
}

//Function of binary to decimal conversion
int binToDec(vector<int> binSeq) {
	int l = binSeq.size();
	int ans = 0;
	for (int i = 0; i < l; i++) {
		ans += binSeq[i] << (l - i - 1);
	}
	return ans;
}

//Functions of energy calculation
//Function used: E(x1, x2, ..., xn) = - 0.5 * sum(1~N) wnm * xn * xm + sum(1~N) tn * xn
int energy(vector<int> state, vector<vector<int>> weight, vector<int> theta) {
	int ans = 0;
	int n = state.size();
	for (int i = 0; i < n; i++) {
		ans += theta[i] * state[i];
		for (int j = 0; j < n; j++) {
			ans += - 0.5 * weight[i][j] * state[i] * state[j];
		}
	}
	return ans;
}
double energyDouble(vector<int> state, vector<vector<double>> weight, vector<double> theta) {
	double ans = 0.0;
	int n = state.size();
	for (int i = 0; i < n; i++) {
		ans += theta[i] * (double) state[i];
		for (int j = 0; j < n; j++) {
			ans += -0.5 * weight[i][j] * (double) state[i] * (double) state[j];
		}
	}
	return ans;
}

//Function to calculate normalisation coefficient A
double normCoefficient(vector<vector<int>> states, double alpha, vector<vector<int>> weight, vector<int> theta) {
	double A = 0.0;
	for (int i = 0; i < states.size(); i++) {
		A += exp(-1.0 * alpha * (double) energy(states[i], weight, theta));
	}
	A = 1.0 / A;
	return A;
}
double normCoefficientDouble(vector<vector<int>> states, double alpha, vector<vector<double>> weight, vector<double> theta) {
	double A = 0.0;
	for (int i = 0; i < states.size(); i++) {
		A += exp(-1.0 * alpha * energyDouble(states[i], weight, theta));
	}
	A = 1.0 / A;
	return A;
}

//Function of Boltzmann distribution calculator
double boltzmannDist(double energy, double norm, double alpha) {
	double y = norm* exp(- alpha * energy);
	return y;
}

//Function of ternary to decimal convertion
int terToDec(int nod, vector<int> digits) {
	int dec = 0;
	for (int i = 0; i < nod; i++) {
		dec += pow(3, nod - 1 - i) * digits[i];
	}
	return dec;
}

//Sigmoid function
double sigmoid(double gain, double sHat) {
	double result;
	result = 1.0 / (1.0 + exp(-1 * gain * sHat));
	return result;
}

//Function of decimal to binary conversion
vector<int> decToBin(int length, int dec) {
	vector<int> bin;
	for (int i = length - 1; i >= 0; i--) {
		int k = dec >> i;
		if (k & 1)
			bin.push_back(1);
		else
			bin.push_back(0);
	}
	return bin;
}

//Function of quaternary to decimal conversion
int quatToDec(int nod, vector<int> digits) {
	int dec = 0;
	for (int i = 0; i < nod; i++) {
		dec += pow(4, nod - 1 - i) * digits[i];
	}
	return dec;
}

int main() {

	srand(time(nullptr));
	
	//--------------------------------------------------------------------------
	//Task 1
	
	//Initialise number of rows and columns of neurons and independent equations
	//3 by 3 neuron matrix, 6 independent equations
	const int ron1 = 3;	const int con1 = 3;	const int noie1 = 6; 
	const int non1 = ron1 * con1; //Number of neurons

	//Provide coefficients of original function to derive energy function
	vector<vector<vector<int>>> xCoefficient1 = vector<vector<vector<int>>>{
		vector<vector<int>>{ vector<int>{ 1, 1, 1 },
							 vector<int>{ 0, 0, 0 },
							 vector<int>{ 0, 0, 0 } },
		vector<vector<int>>{ vector<int>{ 0, 0, 0 },
							 vector<int>{ 1, 1, 1 },
							 vector<int>{ 0, 0, 0 } },
		vector<vector<int>>{ vector<int>{ 0, 0, 0 },
							 vector<int>{ 0, 0, 0 },
							 vector<int>{ 1, 1, 1 } },
		vector<vector<int>>{ vector<int>{ 1, 0, 0 },
							 vector<int>{ 1, 0, 0 },
							 vector<int>{ 1, 0, 0 } },
		vector<vector<int>>{ vector<int>{ 0, 1, 0 },
							 vector<int>{ 0, 1, 0 },
							 vector<int>{ 0, 1, 0 } },
		vector<vector<int>>{ vector<int>{ 0, 0, 1 },
							 vector<int>{ 0, 0, 1 },
							 vector<int>{ 0, 0, 1 } } };
	vector<int> constant1 = vector<int>{ -1, -1, -1, -1, -1, -1 };

	//Given energy equation E(x00,x01,...,x22)=(x00+x01+x02-1)^2+(x10+x12+x13-1)^2+...+(x02+x12+x22-1)^2
	//Note that all indices of neurons above are 0-based for the convenience of programming
	//Calculate connection weights and thresholds
	vector<vector<int>> weight1(non1, vector<int> (non1, 0));
	vector<int> theta1(non1, 0);
	for (int i = 0; i < ron1; i++) {
		for (int j = 0; j < con1; j++) {
			for (int k = 0; k < noie1; k++) {
				theta1[terToDec(2, vector<int> {i, j})] += 
					pow(xCoefficient1[k][i][j], 2) + 2 * constant1[k] * xCoefficient1[k][i][j];
			}
			for (int n = 0; n < ron1; n++) {
				for (int m = 0; m < con1; m++) {
					if (i == n && j == m) {
						weight1[terToDec(2, vector<int> {i, j})][terToDec(2, vector<int> {n, m})] = 0;
					}
					else {
						for (int k = 0; k < noie1; k++) {
							weight1[terToDec(2, vector<int> {i, j})][terToDec(2, vector<int> {n, m})] -= 
								2 * xCoefficient1[k][i][j] * xCoefficient1[k][n][m];
						}
					}
				}
			}
		}
	}

	//Create neuron state vector
	vector<int> state1(non1, 0);

	//Initialise update sequence matrix
	vector<int> updateSeq1(non1);
	iota(updateSeq1.begin(), updateSeq1.end(), 0);

	//Initialise vector containing all possible update sequences
	vector<int> permUpdSeq1;
	do {
		for (auto x : updateSeq1) {
			permUpdSeq1.push_back(x);
		}
	} while (next_permutation(updateSeq1.begin(), updateSeq1.end()));
	
	//Devide permUpdSeq1 into non1! groups in gpdPermUpdSeq1
	int nop1 = permUpdSeq1.size() / non1; //number of permutations
	vector<vector<int>> gpdPermUpdSeq1(nop1);
	for (int i = 0; i < nop1; i++) {
		for (int j = 0; j < non1; j++) {
			gpdPermUpdSeq1[i].push_back(permUpdSeq1[i * non1 + j]);
		}
	}

	//Initialise answer matrix
	vector<vector<int>> answer1;

	//Initialise energy matrix
	vector<vector<int>> energy1(nop1, vector<int> (2 * non1 + 1, 0));
	int energyCounter;

	//Start the loop updating neurons
	int perm1 = 0;
	while (perm1 < nop1) {
		energyCounter = 0;
		//Initialise neuron state
		for (int i = 0; i < ron1; i++) {
			for (int j = 0; j < con1; j++) {
				state1[terToDec(2, vector<int> {i, j})] = rand() % 2;
			}
		}
		energy1[perm1][energyCounter] += energy(state1, weight1, theta1);
		energyCounter++;
		updateSeq1 = gpdPermUpdSeq1[perm1];
		//Update neurons one by one with deterministic and binary model
		for (int n = 0; n < non1; n++) {
			//Compute s^ for neuron with index updateSeq1[n]
			int sHat = 0;
			for (int m = 0; m < non1; m++) {
				sHat += state1[m] * (weight1[m][updateSeq1[n]] + weight1[updateSeq1[n]][m]);
			}
			sHat -= theta1[updateSeq1[n]];
			//Update the neuron with index updateSeq1[n]
			if (sHat > 0) {
				state1[updateSeq1[n]] = 1;
			}
			else {
				state1[updateSeq1[n]] = 0;
			}
			energy1[perm1][energyCounter] += energy(state1, weight1, theta1);
			energyCounter++;
		}

		//Update neurons one by one again again
		random_shuffle(updateSeq1.begin(), updateSeq1.end());
		for (int n = 0; n < non1; n++) {
			int sHat = 0;
			for (int m = 0; m < non1; m++) {
				sHat += state1[m] * (weight1[m][updateSeq1[n]] + weight1[updateSeq1[n]][m]);
			}
			sHat -= theta1[updateSeq1[n]];
			if (sHat > 0) {
				state1[updateSeq1[n]] = 1;
			}
			else {
				state1[updateSeq1[n]] = 0;
			}
			energy1[perm1][energyCounter] += energy(state1, weight1, theta1);
			energyCounter++;
		}
		perm1++;

		if (find(answer1.begin(), answer1.end(), state1) != answer1.end()) {
			//repetitive solution found, do NOTHING
		}
		else {
			answer1.push_back(state1); //new solution found, add to answer matrix
		}
	}

	//Print results for Task 1
	cout << "Task 1 Result----------\n\n";
	cout << "Connection Weights\n\n";
	cout << "wi1\twi2\twi3\twi4\twi5\twi6\twi7\twi8\twi9\n";
	for (int i = 0; i < weight1.size(); i++) {
		for (int j = 0; j < weight1[0].size(); j++) {
			cout << weight1[i][j] << "\t";
		}
		cout << endl;
	}
	cout << "\nThresholds\n";
	cout << "theta1\ttheta2\ttheta3\ttheta4\ttheta5\ttheta6\ttheta7\ttheta8\ttheta9\n";
	for (int i = 0; i < theta1.size(); i++) {
		cout << theta1[i] << "\t";
	}
	cout << "\n\nSolution format\n";
	cout << "x1 x2 x3\nx4 x5 x6\nx7 x8 x9\n\n";
	for (int i = 0; i < answer1.size(); i++) {
		cout << "Solution " << to_string(i + 1) << "\n";
		for (int j = 0; j < non1; j++) {
			if (j % 3 == 2) {
				cout << answer1[i][j] << "\n";
			}
			else {
				cout << answer1[i][j] << " ";
			}

		}
		cout << endl;
	}

	//Create "task_1_energy_changes.csv" file to record energy changes of neuron updates
	ofstream energy1csv;
	energy1csv.open("task_1_energy_changes.csv");
	for (int i = 0; i < energy1[0].size(); i++) {
		energy1csv << i + 1 << ",";
	}
	energy1csv << endl;
	for (int i = 0; i < energy1.size(); i++) {
		for (int j = 0; j < energy1[0].size(); j++) {
			energy1csv << energy1[i][j] << ",";
		}
		energy1csv << endl;
	}
	
	//End of Task 1
	//--------------------------------------------------------------------------
	//Task 2
	
	//Initialise number of neurons and independent equations
	const int non2 = 4; const int noie2 = 4; //4 neurons, 4 independent equations
	
	//Provide coefficients of original function to derive energy function
	vector<vector<int>> xCoefficient2 = { vector<int>{ 1, -1, 2, -1 },
										  vector<int>{ 2, 1, -2, 1 },
										  vector<int>{ -1, 2, 1, 2 },
										  vector<int>{ 0, 1, -1, -1 } };
	vector<int> constant2 = vector<int>{ -3, 0, 0, 1 };

	//Given energy equation E(x0,x1,x2,x3)=(x0+x1+x2+x3-y)^2
	//Calculate connection weights and thresholds
	vector<vector<int>> weight2(non2, vector<int> (non2, 0));
	vector<int> theta2(non2, 0);
	for (int i = 0; i < non2; i++) {
		for (int k = 0; k < noie2; k++) {
			theta2[i] += pow(xCoefficient2[k][i], 2) + 2 * constant2[k] * xCoefficient2[k][i];
		}
		for (int j = 0; j < non2; j++) {
			if (i == j) {
				weight2[i][j] = 0;
			}
			else {
				for (int k = 0; k < noie2; k++) {
					weight2[i][j] -= xCoefficient2[k][i] * xCoefficient2[k][j];
				}
			}
		}
	}

	//Create neuron state vector
	vector<int> state2(non2, 0);

	//Initialise update sequence matrix
	vector<int> updateSeq2(non2);
	iota(updateSeq2.begin(), updateSeq2.end(), 0);

	//Initialise vector containing all possible update sequences
	vector<int> permUpdSeq2;
	do {
		for (auto x : updateSeq2) {
			permUpdSeq2.push_back(x);
		}
	} while (next_permutation(updateSeq2.begin(), updateSeq2.end()));

	//Devide permUpdSeq1 into non1! groups in gpdPermUpdSeq1
	int nop2 = permUpdSeq2.size() / non2; //number of permutations
	vector<vector<int>> gpdPermUpdSeq2(nop2);
	for (int i = 0; i < nop2; i++) {
		for (int j = 0; j < non2; j++) {
			gpdPermUpdSeq2[i].push_back(permUpdSeq2[i * non2 + j]);
		}
	}

	//Initialise answer matrix
	vector<vector<int>> answer2;

	//Initialise energy matrix
	vector<vector<int>> energy2(nop2, vector<int>(2 * non2 + 1, 0));

	//Start the loop updating neurons
	int perm2 = 0;
	while (perm2 < nop2) {
		int energyCounter = 0;
		//Initialise neuron state
		for (int i = 0; i < non2; i++) {
			state2[i] = rand() % 2;
		}
		energy2[perm2][energyCounter] += energy(state2, weight2, theta2);
		energyCounter++;
		updateSeq2 = gpdPermUpdSeq2[perm2];
		//Update neurons one by one with deterministic and binary model
		for (int n = 0; n < non2; n++) {
			//Compute s^ for neuron with index updateSeq2[n]
			int sHat = 0;
			for (int m = 0; m < non2; m++) {
				sHat += state2[m] * (weight2[m][updateSeq2[n]] + weight2[updateSeq2[n]][m]);
			}
			sHat -= theta2[updateSeq2[n]];
			//Update the neuron with index updateSeq2[n]
			if (sHat > 0) {
				state2[updateSeq2[n]] = 1;
			}
			else {
				state2[updateSeq2[n]] = 0;
			}
			energy2[perm2][energyCounter] += energy(state2, weight2, theta2);
			energyCounter++;
		}

		//Update neurons one by one again again
		random_shuffle(updateSeq2.begin(), updateSeq2.end());
		for (int n = 0; n < non2; n++) {
			int sHat = 0;
			for (int m = 0; m < non2; m++) {
				sHat += state2[m] * (weight2[m][updateSeq2[n]] + weight2[updateSeq2[n]][m]);
			}
			sHat -= theta2[updateSeq2[n]];
			if (sHat > 0) {
				state2[updateSeq2[n]] = 1;
			}
			else {
				state2[updateSeq2[n]] = 0;
			}
			energy2[perm2][energyCounter] += energy(state2, weight2, theta2);
			energyCounter++;
		}
		perm2++;

		if (find(answer2.begin(), answer2.end(), state2) != answer2.end()) {
			//repetitive solution found, do NOTHING
		}
		else {
			answer2.push_back(state2); //new solution found, add to answer matrix
		}
	}

	//Print results for Task 2
	cout << "\nTask 2 Result----------\n\n";
	cout << "Connection weights\n\n";
	cout << "wi1\twi2\twi3\twi4\n";
	for (int i = 0; i < weight2.size(); i++) {
		for (int j = 0; j < weight2[0].size(); j++) {
			cout << weight2[i][j] << "\t";
		}
		cout << endl;
	}
	cout << "\nThresholds\n\n";
	cout << "theta1\ttheta2\ttheta3\ttheta4\n";
	for (int i = 0; i < theta2.size(); i++) {
		cout << theta2[i] << "\t";
	}
	cout << "\n\nSolution format";
	cout << "\nx1\tx2\tx3\tx4\n\n";
	for (int i = 0; i < answer2.size(); i++) {
		cout << "Solution " << to_string(i + 1) << "\n";
		for (int j = 0; j < non2; j++) {
			if (j % 4 == 3) {
				cout << answer2[i][j] << "\n";
			}
			else {
				cout << answer2[i][j] << "\t";
			}

		}
		cout << endl;
	}

	//Create "task_2_energy_changes.csv" file to record energy changes of neuron updates
	ofstream energy2csv;
	energy2csv.open("task_2_energy_changes.csv");
	for (int i = 0; i < energy2[0].size(); i++) {
		energy2csv << i + 1 << ",";
	}
	energy2csv << endl;
	for (int i = 0; i < energy2.size(); i++) {
		for (int j = 0; j < energy2[0].size(); j++) {
			energy2csv << energy2[i][j] << ",";
		}
		energy2csv << endl;
	}
	
	//End of Task 2
	//--------------------------------------------------------------------------
	//Task 3
	
	//Initialise number of neurons and independent equations
	const int non3 = 4; const int noie3 = 4; //4 neurons, 4 independent equations

	//Provide coefficients of original function to derive energy function
	vector<vector<int>> xCoefficient3 = { vector<int>{ 1, -1, 1, 1 },
										  vector<int>{ 2, 0, -1, 1 },
										  vector<int>{ 0, 1, -1, -1 },
										  vector<int>{ -1, 1, 1, -1 } };
	vector<int> constant3 = vector<int>{ -3, -2, 2, 1 };

	//Given energy equation E(x0,x1,x2,x3)=(x0+x1+x2+x3-y)^2
	//Calculate connection weights and thresholds
	vector<vector<int>> weight3(non3, vector<int>(non3, 0));
	vector<int> theta3(non3, 0);
	for (int i = 0; i < non3; i++) {
		for (int k = 0; k < noie3; k++) {
			theta3[i] += pow(xCoefficient3[k][i], 2) + 2 * constant3[k] * xCoefficient3[k][i];
		}
		for (int j = 0; j < non3; j++) {
			if (i == j) {
				weight3[i][j] = 0;
			}
			else {
				for (int k = 0; k < noie3; k++) {
					weight3[i][j] -= xCoefficient3[k][i] * xCoefficient3[k][j];
				}
			}
		}
	}

	//Initialise neuron state vector
	vector<int> state3(non3, 0);

	//Initialise update sequence matrix
	vector<int> updateSeq3(non3);
	iota(updateSeq3.begin(), updateSeq3.end(), 0);

	//Initialise iteration counters
	int copy3 = 0;
	int itr3 = 0;

	//Initialise matrices to store number of copies of all states
	vector<int> noc3_10k(pow(2, non3), 0);
	vector<int> noc3_50k(pow(2, non3), 0);
	vector<int> noc3_100k(pow(2, non3), 0);

	cout << "\nTask 3 Result----------\n\n";

	//Gain
	double alpha3;
	cout << "Specify alpha as a double, alpha = ";
	cin >> alpha3;
	cout << "\n";


	//Start the loop updating neurons
	while (copy3 < 1000) {//1000 Gibbs copies
		//Initialise neuron state
		for (int i = 0; i < non3; i++) {
			state3[i] = rand() % 2;
		}
		itr3 = 0;
		while (itr3 < 100000) {
			//Update neurons one by one with deterministic and binary model
			for (int n = 0; n < non3; n++) {
				//Compute s^ for neuron with index updateSeq3[n]
				int sHat = 0;
				for (int m = 0; m < non3; m++) {
					sHat += state3[m] * (weight3[m][updateSeq3[n]] + weight3[updateSeq3[n]][m]);
				}
				sHat -= theta3[updateSeq3[n]];
				//Update the neuron with index updateSeq3[n]
				if (rand() <= (double) RAND_MAX * sigmoid(alpha3, (double) sHat)) {
					state3[updateSeq3[n]] = 1;
					itr3++;
				}
				else {
					state3[updateSeq3[n]] = 0;
					itr3++;
				}
				switch (itr3) {
				case 9999:
					noc3_10k[binToDec(state3)]++;
					break;
				case 49999:
					noc3_50k[binToDec(state3)]++;
					break;
				case 99999:
					noc3_100k[binToDec(state3)]++;
					break;
				}
			}
		}
		copy3++;
	}

	//List all possible states
	vector<vector<int>> states3 = { vector<int>{ 0, 0, 0, 0 }, vector<int>{ 0, 0, 0, 1 }, 
									vector<int>{ 0, 0, 1, 0 }, vector<int>{ 0, 0, 1, 1 },
									vector<int>{ 0, 1, 0, 0 }, vector<int>{ 0, 1, 0, 1 },
									vector<int>{ 0, 1, 1, 0 }, vector<int>{ 0, 1, 1, 1 },
									vector<int>{ 1, 0, 0, 0 }, vector<int>{ 1, 0, 0, 1 },
									vector<int>{ 1, 0, 1, 0 }, vector<int>{ 1, 0, 1, 1 },
									vector<int>{ 1, 1, 0, 0 }, vector<int>{ 1, 1, 0, 1 },
									vector<int>{ 1, 1, 1, 0 }, vector<int>{ 1, 1, 1, 1 } };

	//Compute Boltzmann distribution
	vector<double> boltzmannDistNum3(pow(2, non3), 0.0);
	double normCoeff3 = normCoefficient(states3, alpha3, weight3, theta3);
	vector<double> energy3(pow(2, non3), 0.0);
	for (int i = 0; i < boltzmannDistNum3.size(); i++) {
		energy3[i] += (double) energy(states3[i], weight3, theta3);
		boltzmannDistNum3[binToDec(states3[i])] = 1000.0 * boltzmannDist(energy3[i], normCoeff3, alpha3);
	}
	vector<int> boltzmannDistNum3Int(pow(2, non3), 0);
	for (int i = 0; i < boltzmannDistNum3.size(); i++) {
		boltzmannDistNum3Int[i] += (int) (boltzmannDistNum3[i] + 0.5);
	}

	//Print results for Task 3
	cout << "Connection weights\n\n";
	cout << "wi1\twi2\twi3\twi4\n";
	for (int i = 0; i < weight3.size(); i++) {
		for (int j = 0; j < weight3[0].size(); j++) {
			cout << weight3[i][j] << "\t";
		}
		cout << endl;
	}
	cout << "\nThresholds\n\n";
	cout << "theta1\ttheta2\ttheta3\ttheta4\n";
	for (int i = 0; i < theta3.size(); i++) {
		cout << theta3[i] << "\t";
	}
	cout << "\n\nNote that all states are represented by their decimal equivalents, e.g. 0101->5\n\n";
	cout << "State (binary equivalent)\t# of occurences (10k)\t# of occurences (50k)\t";
	cout << "# of occurences (100k)\t# of occurences (Boltzmann)" << endl;
	//Print out distribution
	for (int i = 0; i < pow(2, non3); i++) {
		cout << i << "\t\t\t\t" << noc3_10k[i] << "\t\t\t" << noc3_50k[i] << "\t\t\t";
		cout << noc3_100k[i] << "\t\t\t" << boltzmannDistNum3Int[i] << endl;
	}

	//Output distribution to "task_3_distribution.csv" file
	ofstream dist3csv;
	dist3csv.open("task_3_distribution.csv");
	for (int i = 0; i < pow(2, non3); i++) {
		dist3csv << i << "," << noc3_10k[i] << "," << noc3_50k[i] << ",";
		dist3csv << noc3_100k[i] << "," << boltzmannDistNum3Int[i] << endl;
	}
	
	//End of Task 3
	//--------------------------------------------------------------------------
	//Task 4
	
	//Initialise number of rows and columns of neurons and independent equations
	const int ron4 = ron1;	const int con4 = con1;	const int noie4 = noie1;
	const int non4 = non1; //Number of neurons

	//Use connection weights and threholds calculated in Task 1
	vector<vector<int>> weight4 = weight1;
	vector<int> theta4 = theta1;

	//Create neuron state vector
	vector<int> state4(non4, 0);

	//Initialise update sequence matrix
	vector<int> updateSeq4(non4);
	iota(updateSeq4.begin(), updateSeq4.end(), 0);

	//Initialise iteration counters
	int copy4 = 0;
	int itr4 = 0;

	//Initialise matrices to store number of copies of all states
	vector<int> noc4_10k(pow(2, non4), 0);
	vector<int> noc4_50k(pow(2, non4), 0);
	vector<int> noc4_100k(pow(2, non4), 0);

	cout << "\n\nTask 4 Result----------\n\n";

	//Gain
	double alpha4;
	cout << "Specify alpha as a double, alpha = ";
	cin >> alpha4;
	cout << "\n";

	//Start the loop updating neurons
	while (copy4 < 1000) {//1000 Gibbs copies
		//Initialise neuron state
		for (int i = 0; i < non4; i++) {
			state4[i] = rand() % 2;
		}
		itr4 = 0;
		while (itr4 < 100000) {
			//Update neurons one by one with deterministic and binary model
			for (int n = 0; n < non4; n++) {
				//Compute s^ for neuron with index updateSeq3[n]
				int sHat = 0;
				for (int m = 0; m < non4; m++) {
					sHat += state4[m] * (weight4[m][updateSeq4[n]] + weight4[updateSeq4[n]][m]);
				}
				sHat -= theta4[updateSeq4[n]];
				//Update the neuron with index updateSeq3[n]
				if (rand() <= (double)RAND_MAX * sigmoid(alpha4, (double)sHat)) {
					state4[updateSeq4[n]] = 1;
					itr4++;
				}
				else {
					state4[updateSeq4[n]] = 0;
					itr4++;
				}
				switch (itr4) {
				case 9999:
					noc4_10k[binToDec(state4)]++;
					break;
				case 49999:
					noc4_50k[binToDec(state4)]++;
					break;
				case 99999:
					noc4_100k[binToDec(state4)]++;
					break;
				}
			}
		}
		copy4++;
	}

	//List all possible states
	vector<vector<int>> states4(pow(2, non4), vector<int>(non4));
	vector<int> states4dec(pow(2, non4));
	iota(states4dec.begin(), states4dec.end(), 0);
	for (int i = 0; i < states4dec.size(); i++) {
		states4[i] = decToBin(non4, states4dec[i]);
	}

	//Compute Boltzmann distribution
	vector<double> boltzmannDistNum4(pow(2, non4), 0.0);
	double normCoeff4 = normCoefficient(states4, alpha4, weight4, theta4);
	vector<double> energy4(pow(2, non4), 0.0);
	for (int i = 0; i < pow(2, non4); i++) {
		energy4[i] += (double) energy(states4[i], weight4, theta4);
		boltzmannDistNum4[binToDec(states4[i])] = 1000.0 * boltzmannDist(energy4[i], normCoeff4, alpha4);
	}
	vector<int> boltzmannDistNum4Int(pow(2, non4), 0);
	for (int i = 0; i < boltzmannDistNum4.size(); i++) {
		boltzmannDistNum4Int[i] += (int)(boltzmannDistNum4[i] + 0.5);
	}

	//Print results for Task 4
	cout << "Note that all states are represented by their decimal equivalents, e.g. 0101->5\n\n";
	cout << "State (binary equivalent)\t# of occurences (10k)\t# of occurences (50k)\t";
	cout << "# of occurences (100k)\t# of occurences (Boltzmann)" << endl;
	//Print out distribution
	for (int i = 0; i < pow(2, non4); i++) {
		cout << i << "\t\t\t\t" << noc4_10k[i] << "\t\t\t" << noc4_50k[i] << "\t\t\t";
		cout << noc4_100k[i] << "\t\t\t" << boltzmannDistNum4Int[i] << endl;
	}

	//Output distribution to "task_4_distribution.csv" file
	ofstream dist4csv;
	dist4csv.open("task_4_distribution.csv");
	for (int i = 0; i < pow(2, non4); i++) {
		dist4csv << i << "," << noc4_10k[i] << "," << noc4_50k[i] << ",";
		dist4csv << noc4_100k[i] << "," << boltzmannDistNum4Int[i] << endl;
	}
	
	//End of Task 4
	//--------------------------------------------------------------------------
	//Task 5
	
	//Arrangements of 4 cities (graphical representations not to scale)
	
	//Arrangement 1 (Sides 10*10)
	//City 1----------City 2
	//|					|
	//|					|
	//|					|
	//|					|
	//|					|
	//City 4----------City 3

	//Arrangement 2 (Distance intervals 10, 10, 10, 10)
	//City 1---City 2---City 3---City4

	//Arrangement 3 (Triangle base*height 6*4)
	//City 1---City 2---City 3
	//	\		|		/
	//	 \		|	   /
	//	  \		|	  /
	//	   \	|	 /
	//		\	|	/
	//		 \	|  /
	//		  \ | /
	//		  City 4

	//Arrangement 4 (Parallelogram sides 5 by 6, City 2-4 distance 5)
	//City 1-----------City 2
	//	\				\
	//	 \				 \
	//	  \				  \
	//	   \			   \
	//	  City 4-----------City 3

	//Distance table for the 4 arrangements
	vector<vector<vector<double>>> distanceTable = vector<vector<vector<double>>>{ 
		vector<vector<double>>{vector<double>{ 0.0,			10.0,		sqrt(20.0),	10.0		},
							   vector<double>{ 10.0,		0.0,		10.0,		sqrt(20.0)	},
							   vector<double>{ sqrt(20.0),	10.0,		0.0,		10.0		},
							   vector<double>{ 10.0,		sqrt(20.0),	10.0,		0.0			} },
		vector<vector<double>>{vector<double>{ 0.0,			10.0,		20.0,		30.0		},
							   vector<double>{ 10.0,		0.0,		10.0,		20.0		},
							   vector<double>{ 20.0,		10.0,		0.0,		10.0		},
							   vector<double>{ 30.0,		20.0,		10.0,		0.0			} },
		vector<vector<double>>{vector<double>{ 0.0,			3.0,		6.0,		5.0			},
							   vector<double>{ 3.0,			0.0,		3.0,		4.0			},
							   vector<double>{ 6.0,			3.0,		0.0,		5.0			},
							   vector<double>{ 5.0,			4.0,		5.0,		0.0			} },
		vector<vector<double>>{vector<double>{ 0.0,			6.0,		sqrt(97.0),	5.0			},
							   vector<double>{ 6.0,			0.0,		5.0,		5.0			},
							   vector<double>{ sqrt(97.0),	5.0,		0.0,		6.0			},
							   vector<double>{ 5.0,			5.0,		6.0,		0.0			} } };

	cout << "\n\nTask 5 Result----------\n\n";

	//Specify which arrangement to use
	int arrangement;
	cout << "Arrangement 1 (Sides 10 * 10)\n";
	cout << "City 1----------City 2\n";
	cout << "|                  |\n";
	cout << "|                  |\n";
	cout << "|                  |\n";
	cout << "|                  |\n";
	cout << "|                  |\n";
	cout << "City 4----------City 3\n\n";
	cout << "Arrangement 2 (Distance intervals 10, 10, 10, 10)\n";
	cout << "City 1---City 2---City 3---City4\n\n";
	cout << "Arrangement 3 (Triangle base*height 6*4)\n";
	cout << "City 1---City 2---City 3\n";
	cout << "   \\       |       /\n";
	cout << "    \\      |      /\n";
	cout << "     \\     |     /\n";
	cout << "      \\    |    /\n";
	cout << "       \\   |   /\n";
	cout << "        \\  |  /\n";
	cout << "         \\ | /\n";
	cout << "        City 4\n\n";
	cout << "Arrangement 4 (Parallelogram sides 5 by 6, City 2-4 distance 5)\n";
	cout << "City 1-----------City 2\n";
	cout << "   \\               \\\n";
	cout << "    \\               \\\n";
	cout << "     \\               \\\n";
	cout << "      \\               \\\n";
	cout << "     City 4-----------City 3\n\n";
	cout << "Please choose from Arrangements 1-4: ";
	cin >> arrangement;
	arrangement--;

	//Specify gain and weight for Ec
	double alpha5;
	cout << "Specify alpha as a double, alpha = ";
	cin >> alpha5;
	double beta5;
	cout << "Specify beta as a double, beta = ";
	cin >> beta5;
	cout << "\n";

	//Initialise number of neurons
	const int ron5 = 4; const int con5 = 4; const int noie5 = 4;
	const int non5 = ron5 * con5;

	//Initialise given info to compute thresholds and connection weights
	vector<vector<vector<int>>> xCoefficient5Ec = vector<vector<vector<int>>>{ 
		vector<vector<int>>{ vector<int>{ 1, 1, 1, 1 },
							 vector<int>{ 0, 0, 0, 0 },
							 vector<int>{ 0, 0, 0, 0 },
							 vector<int>{ 0, 0, 0, 0 } },
		vector<vector<int>>{ vector<int>{ 0, 0, 0, 0 },
							 vector<int>{ 1, 1, 1, 1 },
							 vector<int>{ 0, 0, 0, 0 },
							 vector<int>{ 0, 0, 0, 0 } },
		vector<vector<int>>{ vector<int>{ 0, 0, 0, 0 },
							 vector<int>{ 0, 0, 0, 0 },
							 vector<int>{ 1, 1, 1, 1 },
							 vector<int>{ 0, 0, 0, 0 } },
		vector<vector<int>>{ vector<int>{ 0, 0, 0, 0 },
							 vector<int>{ 0, 0, 0, 0 },
							 vector<int>{ 0, 0, 0, 0 },
							 vector<int>{ 1, 1, 1, 1 } },
		vector<vector<int>>{ vector<int>{ 1, 0, 0, 0 },
							 vector<int>{ 1, 0, 0, 0 },
							 vector<int>{ 1, 0, 0, 0 },
							 vector<int>{ 1, 0, 0, 0 } },
		vector<vector<int>>{ vector<int>{ 0, 1, 0, 0 },
							 vector<int>{ 0, 1, 0, 0 },
							 vector<int>{ 0, 1, 0, 0 },
							 vector<int>{ 0, 1, 0, 0 } },
		vector<vector<int>>{ vector<int>{ 0, 0, 1, 0 },
							 vector<int>{ 0, 0, 1, 0 },
							 vector<int>{ 0, 0, 1, 0 },
							 vector<int>{ 0, 0, 1, 0 } },
		vector<vector<int>>{ vector<int>{ 0, 0, 0, 1 },
							 vector<int>{ 0, 0, 0, 1 },
							 vector<int>{ 0, 0, 0, 1 },
							 vector<int>{ 0, 0, 0, 1 } } };
	vector<int> constant5Ec = vector<int>{ -1, -1, -1, -1 };

	//Calculate connection weights and thresholds
	vector<vector<double>> weight5(non5, vector<double>(non5, 0.0));
	vector<double> theta5(non5, 0.0);
	//Ec
	for (int i = 0; i < ron5; i++) {
		for (int j = 0; j < con5; j++) {
			for (int k = 0; k < noie5; k++) {
				theta5[quatToDec(2, vector<int> {i, j})] += 
					beta5 * (double) (pow(xCoefficient5Ec[k][i][j], 2) + 2 * constant5Ec[k] * xCoefficient5Ec[k][i][j]);
			}
			for (int n = 0; n < ron5; n++) {
				for (int m = 0; m < con5; m++) {
					if (i == n && j == m) {
						weight5[quatToDec(2, vector<int> {i, j})][quatToDec(2, vector<int> {n, m})] = 0.0;
					}
					else {
						for (int k = 0; k < noie5; k++) {
							weight5[quatToDec(2, vector<int> {i, j})][quatToDec(2, vector<int> {n, m})] -= 
								beta5 * (double) (2 * xCoefficient5Ec[k][i][j] * xCoefficient5Ec[k][n][m]);
						}
					}
				}
			}
		}
	}
	//El
	for (int i = 0; i < ron5; i++) {
		for (int j = 0; j < con5; j++) {
			for (int k = 0; k < noie5; k++) {
				if (i + 1 >= 4) {
					weight5[quatToDec(2, vector<int>{ i, j })][quatToDec(2, vector<int>{ i + 1 - 4, k })] += 
						distanceTable[arrangement][j][k];
				}
				else {
					weight5[quatToDec(2, vector<int>{ i, j })][quatToDec(2, vector<int>{ i + 1, k })] += 
						distanceTable[arrangement][j][k];
				}
				
			}
		}
	}
	
	//Initialise neuron state vector
	vector<int> state5(non5, 0);

	//Initialise update sequence matrix
	vector<int> updateSeq5(non5);
	iota(updateSeq5.begin(), updateSeq5.end(), 0);

	//Initialise iteration counters
	int copy5 = 0;
	int itr5 = 0;

	//Initialise vector to store number of copies of all states
	vector<int> noc5(pow(2, non5), 0);

	//Start the loop updating neurons
	while (copy5 < 1000000) {//1 million Gibbs copies
		//Initialise neuron state
		for (int i = 0; i < non5; i++) {
			state5[i] = rand() % 2;
		}
		itr5 = 0;
		while (itr5 < 100) {//100 updates for each Gibbs copy
			//Update neurons one by one with deterministic and binary model
			for (int n = 0; n < non5; n++) {
				//Compute s^ for neuron with index updateSeq3[n]
				int sHat = 0;
				for (int m = 0; m < non5; m++) {
					sHat += state5[m] * (weight5[m][updateSeq5[n]] + weight5[updateSeq5[n]][m]);
				}
				sHat -= theta5[updateSeq5[n]];
				//Update the neuron with index updateSeq3[n]
				if (rand() <= (double)RAND_MAX * sigmoid(alpha5, (double)sHat)) {
					state5[updateSeq5[n]] = 1;
					itr5++;
				}
				else {
					state5[updateSeq5[n]] = 0;
					itr5++;
				}
			}
		}
		noc5[binToDec(state5)]++;
		copy5++;
	}

	//List all possible states
	vector<vector<int>> states5(pow(2, non5), vector<int>(non5));
	vector<int> states5dec(pow(2, non5));
	iota(states5dec.begin(), states5dec.end(), 0);
	for (int i = 0; i < states5dec.size(); i++) {
		states5[i] = decToBin(non5, states5dec[i]);
	}

	//Compute Boltzmann distribution
	vector<double> boltzmannDistNum5(pow(2, non5), 0.0);
	double normCoeff5 = normCoefficientDouble(states5, alpha5, weight5, theta5);
	vector<double> energy5(pow(2, non5), 0.0);
	for (int i = 0; i < pow(2, non5); i++) {
		energy5[i] += energyDouble(states5[i], weight5, theta5);
		boltzmannDistNum5[binToDec(states5[i])] = 1000000.0 * boltzmannDist(energy5[i], normCoeff5, alpha5);
	}
	vector<int> boltzmannDistNum5Int(pow(2, non5), 0);
	for (int i = 0; i < boltzmannDistNum5.size(); i++) {
		boltzmannDistNum5Int[i] += (int)(boltzmannDistNum5[i] + 0.5);
	}

	//List top 10 max
	vector<int> noc5Copy = noc5;
	vector<vector<int>> topTen(2, vector<int>(10, 0));
	for (int i = 0; i < 10; i++) {
		topTen[1][i] += *max_element(noc5Copy.begin(), noc5Copy.end());
		for (int j = 0; j < noc5.size(); j++) {
			if (topTen[1][i] == noc5[j]) {
				topTen[0][i] = j;
			}
		}
		for (int k = 0; k < noc5Copy.size(); k++) {
			if (topTen[1][i] == noc5Copy[k]) {
				noc5Copy.erase(noc5Copy.begin() + k);
			}
		}
	}

	//Print results for Task 5
	cout << "Top 10 states with most occurences\n\n";
	cout << "State (binary equivalent)\t# of occurences (100)\t# of occurences (Boltzmann)" << endl;
	//Print out distribution
	ofstream t5top10csv;
	t5top10csv.open("task_5_top10.csv");
	for (int i = 0; i < 10; i++) {
		cout << topTen[0][i]  << "\t\t\t\t" << topTen[1][i] << "\t\t\t" << boltzmannDistNum5Int[topTen[0][i]] << endl;
		t5top10csv << topTen[0][i] << "," << topTen[1][i] << "," << boltzmannDistNum5Int[topTen[0][i]] << endl;
	}

	//End of Task 5
	
	return 0;

}