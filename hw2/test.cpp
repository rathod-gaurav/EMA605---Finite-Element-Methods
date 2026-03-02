#include <iostream>
#include <Eigen/Dense>
using namespace std;



int main(){
    Eigen::MatrixXd Kglobal(10,10);
    Eigen::VectorXd Fglobal(10);

    Kglobal(0,0) = 1;
    Kglobal(2,3) = 2;
    Fglobal(2) = 0.1;

    cout << Kglobal(0,0) << endl;
    cout << Kglobal(2,3) << endl;
    cout << Fglobal[2] << endl;

    Eigen::VectorXd D = Kglobal.partialPivLu().solve(Fglobal);
}