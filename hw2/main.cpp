//1D problem

#include <iostream>
#include <vector>
#include <iomanip> //for setprecision function during cout
#include <cmath>
#include <stdexcept>
#include<functional>
#include <fstream> //to store datapoints for plot
#include <string>
#include <Eigen/Dense>

using namespace std;

template<int Nne>
struct Segment{ //struct to represent start and end GLOBAL NODE NUMBERS and COORDINATES of a element subdomain
    int node[Nne]; // GLOBAL NODE NUMBERS
    double x[Nne]; // GLOBAL COORDINATES CORRESPONDING TO THE NODES
};

struct QuadratureRule {
    std::vector<double> points;
    std::vector<double> weights;
};

QuadratureRule gauss_legendre(unsigned int n) {
    QuadratureRule rule;

    switch(n) {
        case 1:
            rule.points  = { 0.0 };
            rule.weights = { 2.0 };
            break;

        case 2:
            rule.points  = { -0.5773502691896257,  0.5773502691896257 };
            rule.weights = {  1.0,                 1.0 };
            break;

        case 3:
            rule.points  = { -0.7745966692414834, 0.0, 0.7745966692414834 };
            rule.weights = {  0.5555555555555556, 0.8888888888888888, 0.5555555555555556 };
            break;

        case 4:
            rule.points  = { -0.8611363115940526, -0.3399810435848563,
                              0.3399810435848563,  0.8611363115940526 };
            rule.weights = {  0.3478548451374539,  0.6521451548625461,
                              0.6521451548625461,  0.3478548451374539 };
            break;

        case 5:
            rule.points  = { -0.9061798459386640, -0.5384693101056831,
                              0.0,
                              0.5384693101056831,  0.9061798459386640 };
            rule.weights = {  0.2369268850561891,  0.4786286704993665,
                              0.5688888888888889,  0.4786286704993665,
                              0.2369268850561891 };
            break;

        case 6:
            rule.points  = { -0.9324695142031521, -0.6612093864662645,
                             -0.2386191860831969,  0.2386191860831969,
                              0.6612093864662645,  0.9324695142031521 };
            rule.weights = {  0.1713244923791704,  0.3607615730481386,
                              0.4679139345726910,  0.4679139345726910,
                              0.3607615730481386,  0.1713244923791704 };
            break;

        default:
            throw std::invalid_argument("Gauss-Legendre quadrature not implemented for this n");
    }

    return rule;
}

int main(){    
    double ll = 0.0; //lower limit of domain
    double ul = 0.1; //upper limit of domain
    enum class Problem{i,ii,iii,iv};
    enum class BCType{Dirischlet, Neumann};
    
    //EDIT THIS
    int Nel = 3; //number of elements in the domain
    constexpr int Nne = 2; //number of nodes in the element
    int quadRule = 2; //number of quardature points to consider for numerical integration
    Problem p = Problem::i;
    BCType bc1, bc2;
    
    int Nt = Nel + 1 + (Nne -2)*Nel; //total number of nodes in the domain
    //variables for boundary conditions
    double g1 = 0.0;
    int nodeLocation1 = 0;
    double g2 = 1e-3;
    double h = 1e6;
    int nodeLocation2 = Nt - 1;
    
    using Segment1D = Segment<Nne>; //resuable alias for 1D segment
    double he = (ul-ll)/Nel; //node spacing in an element
    double Area = 1e-4;
    double E = 1e11;
    double ft;
    std::function<double(double)> f;

    if(p == Problem::i){
        bc1 = BCType::Dirischlet;
        bc2 = BCType::Dirischlet; //declare BCns here
        ft = 0;
        f = [ft](double x = 0.0){
            return 0;
        };
    }
    else if(p == Problem::ii){
        bc1 = BCType::Dirischlet;
        bc2 = BCType::Dirischlet; //declare BCns here
        ft = 1e6;
        f = [ft](double x = 0.0){
            return ft;
        };
    }
    else if(p == Problem::iii){
        bc1 = BCType::Dirischlet; 
        bc2 = BCType::Neumann; //declare BCns here
        ft = 1e6;
        f = [ft](double x = 0.0){
            return ft;
        };
    }
    else if(p == Problem::iv){
        bc1 = BCType::Dirischlet;
        bc2 = BCType::Dirischlet; //declare BCns here
        ft = 1e7;
        f = [ft](double x = 0.0){
            return ft*x;
        };
    }

    vector<double> x(Nt); // x coordinates of nodes
    for(int i = 0; i < Nt ; i++){
        x[i] = ll + i*(he/(Nne-1));
    }

    vector<Segment1D> element; // element vector storing global node numbering and locations
    element.reserve(Nel);
    for(int e = 0; e < Nel ; e++){
        Segment1D seg;
        for(int j = 0; j < Nne ; j++){
            seg.node[j] = e + j + e*(Nne - 2);
            seg.x[j] = x[e + j + e*(Nne - 2)];
        }
        element.push_back(seg);
    }
    

    //debugging block start
    // for (const auto& seg : element) {
    //     cout << seg.n[0] << " ; " << seg.n[Nne-1] << endl;
    //     cout << seg.node[0] << " -> " << seg.x[0] << " ; " << seg.node[Nne-1] << " -> " << seg.x[Nne-1] << endl;
    // }
    // cout << element[Nel-1].node[Nne-1] << endl;
    // cout << Nt << endl;

    // cout << fixed << setprecision(6) << f(element[1].x[2]) << endl;
    //debugging block end

    auto xi_at_node = [Nne](unsigned int node){ //function to find value of xi at any node in the element
        double xi;

        if(node == 0){
            xi = -1.;
        }
        else if(node == Nne-1){
            xi = 1.;
        }
        else if(node <= Nne-1){
            xi = -1. + 2.*node/(Nne-1);
        }
        else{
            std::cout << "Error: you input node number is "
                << node << " but there are only " 
                << Nne << " nodes in an element.\n";
            exit(0);
        }
        return xi;
    };

    auto basis_function = [Nne, xi_at_node](unsigned int node, double xi){ //function to calculate value of basis function for a given node at given location xi
        double value = 1.; //Store the value of the basis function in this variable

        for (unsigned int i = 0; i < Nne; i++){
            if (i != node){
                value = value * (xi - xi_at_node(i))/(xi_at_node(node) - xi_at_node(i));
            }
        }

        return value;
    };

    auto basis_gradient = [Nne, xi_at_node, basis_function](unsigned int node, double xi){ //function to calculate value of gradient of basis function for a given node at given location xi

        double tol = 1e-12; //tolerance

        //EDIT
        // Check if xi is at a node
        for(unsigned int i = 0; i < Nne; i++){
            if (std::abs(xi - xi_at_node(i)) < tol) {
                // special formula at node
                if(i == node) {
                    double sum = 0.0;
                    for(unsigned int j = 0; j < Nne; j++){
                        if(j != node)
                            sum += 1.0/(xi_at_node(node) - xi_at_node(j));
                    }
                    return sum;
                } else {
                    double prod = 1.0;
                    for(unsigned int j = 0; j < Nne; j++){
                        if(j != node && j != i)
                            prod *= (xi_at_node(i)-xi_at_node(j)) /
                                    (xi_at_node(node)-xi_at_node(j));
                    }
                    return prod / (xi_at_node(node) - xi_at_node(i));
                }
            }
        }

        // Otherwise: compact formula (safe away from nodes)
        double sum = 0.0;
        for(unsigned int j = 0; j < Nne; j++){
            if(j != node)
                sum += 1.0 / (xi - xi_at_node(j));
        }

        return basis_function(node, xi) * sum;
    };


    //debugging block start
    // cout << xi_at_node(0) << endl;
    // cout << basis_function(1,0.5) << endl;
    // cout << basis_gradient(1,0.12) << endl;
    //debugging block end

    QuadratureRule q = gauss_legendre(quadRule);
    std::vector<double> quad_points, quad_weights; //vectors to store Gaussian quadrature points and their corresponding weights
    quad_points.resize(quadRule); quad_weights.resize(quadRule);
    quad_points = q.points;
    quad_weights = q.weights;

    auto global_x_from_xi = [element, Nne, basis_function](int e, double qi){//function to obtain global x coordinate given a quadrature point qi for element e
        double x = 0.0;
        for(int A = 0; A < Nne ; A++){
            x += basis_function(A,qi)*element[e].x[A];
        }
        return x;
    };

    //debugging block start
    // for(const auto& point : quad_points){
    //     cout << point << endl;
    // }    
    // cout << element[Nel-1].x[Nne-1] << endl;
    // cout << global_x_from_xi(Nel-1, 1) << endl;

    // double integral_value = 0.;
    // for(int i = 0 ; i < quadRule ; i++){
    //     integral_value += basis_function(1,quad_points[i])*quad_weights[i]; 
    // }
    // cout << integral_value << endl;
    //debugging block end

    // Assembly Process

    Eigen::MatrixXd Kglobal(Nt,Nt);
    Eigen::VectorXd Fglobal(Nt);
    for(int e = 0 ; e < Nel ; e++){
        std::vector<std::vector<double>> Klocal(Nne , std::vector<double>(Nne, 0.0));
        std::vector<double> Flocal(Nne , 0.0);
        for(int A = 0 ; A < Nne ; A++){
            for(int B = 0 ; B < Nne ; B++){
                for(int i = 0 ; i < quadRule ; i++){
                    Klocal[A][B] += basis_gradient(A,quad_points[i])*((2*E*Area)/he)*basis_gradient(B,quad_points[i])*quad_weights[i];
                }
                Kglobal(e*(Nne - 1) + A , e*(Nne - 1) + B) += Klocal[A][B];
            }
            for(int i = 0 ;  i < quadRule ; i++){
                Flocal[A] += basis_function(A,quad_points[i])*f(global_x_from_xi(e,quad_points[i]))*(he/2)*quad_weights[i];
            }
            Fglobal(e*(Nne - 1) + A) += Flocal[A];
        }
        // debugging block start
        // cout << "rows : " << Klocal.size() << endl;
        // cout << "columns : " << Klocal[0].size() << endl;
        // cout << "" << endl;
        // for (const auto& row : Klocal) {
        //     for (double val : row) {
        //         std::cout << val << " ";
        //     }
        //     std::cout << '\n';
        // }
        // debugging block end
    }

    //debugging block start
    // cout << "Kglobal : " << endl;
    // cout << "rows : " << Kglobal.size() << endl;
    // cout << "columns : " << Kglobal[0].size() << endl;
    // cout << "" << endl;
    // for (const auto& row : Kglobal) {
    //     for (double val : row) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << '\n';
    // }

    // // cout << "special entry : " << Kglobal[1][0] << endl;

    // cout << "\n";
    // cout << "rows : " << Fglobal.size() << endl;
    // cout << "columns : " << 1 << endl;
    // for (double val : Fglobal) {
    //     std::cout << val << " ";
    // }
    // std::cout << '\n';
    //debugging block end

    //Check and Apply Boundary Conditions
    Eigen::MatrixXd K_reduced;
    Eigen::VectorXd F_reduced;

    if(bc1 == BCType::Dirischlet && bc2 == BCType::Dirischlet){
        // Extract the inner block (rows 1 to n-2, cols 1 to n-2)
        K_reduced = Kglobal.block(1, 1, Nt-2, Nt-2);
        F_reduced = Fglobal.segment(1, Nt-2);

        // Apply Boundary Conditions
        F_reduced -= Kglobal.block(1, 0, Nt-2, 1) * g1;     // Column 0 (first)
        F_reduced -= Kglobal.block(1, Nt-1, Nt-2, 1) * g2;   // Column n-1 (last)
    }
    else if(bc1 == BCType::Dirischlet && bc2 == BCType::Neumann){
        // remove first row and first column of Kglobal and remove first row of Fglobal
        K_reduced = Kglobal.block(1,1,Nt-1,Nt-1);
        F_reduced = Fglobal.segment(1,Nt-1);

        //apply boundary conditions
        F_reduced -= Kglobal.block(1,0,Nt-1,1)*g1;
        F_reduced(F_reduced.rows() - 1) += h;

    }
    else if(bc1 == BCType::Neumann && bc2 == BCType::Dirischlet){
        // remove last row and last column of Kglobal and remove last row of Fglobal
        K_reduced = Kglobal.block(0,0,Nt-1,Nt-1);
        F_reduced = Fglobal.segment(0,Nt-1);

        //apply boundary conditions
        F_reduced -= Kglobal.block(0,Nt-1,1,Nt-1)*g1;
        F_reduced(0) += h;
    }
    else{
        cout << "Incorrect Boundary Conditions - cannot be Neumann at both ends" << endl;
    }
    
    //debugging block start
    // cout << "BCns Applied : obtained final Kglobal and Fglobal" << endl;
    // cout << Kglobal.size() << " , " << Kglobal[0].size() << endl;
    // cout << Fglobal.size() << endl;
    //debugging block end

    Eigen::VectorXd D_reduced = K_reduced.partialPivLu().solve(F_reduced);

    //debugging block start
    // cout << "solution obtained" << endl;
    // for(double& val : D){
    //     cout << val << endl;
    // }
    //debugging block end

    // Final solution
    Eigen::VectorXd D(Nt);
    if(bc1 == BCType::Dirischlet && bc2 == BCType::Dirischlet){
        D << g1 , D_reduced , g2;
    }
    else if(bc1 == BCType::Dirischlet && bc2 == BCType::Neumann){
        D << g1 , D_reduced;
    }
    else if(bc1 == BCType::Neumann && bc2 == BCType::Dirischlet){
        D << D_reduced , g2;
    }
    //debugging block start
    cout << "Final Solution" << endl;
    for(double& val : D){
        cout << std::setprecision(16) << val << endl;
    }
    //debugging block end

    // Analytical Solutions
    
    auto u_analytical_x = [g1, g2, Area, E, ft, ul, h, p](double x){
        if(p == Problem::i){
            return ((g2-g1)/ul)*x + g1;
        }
        else if(p == Problem::ii){ 
            return (-1/(Area*E))*(ft*0.5*x*x) + ((g2-g1)/ul)*x + g1 + ((ft*ul)/(2*Area*E))*x;
        }
        else if(p == Problem::iii){
            return (-1/(Area*E))*(ft*0.5*x*x) + (x/(Area*E))*(h + ft*ul) + g1;
        }
        else if(p == Problem::iv){
            return (-1/(6*Area*E))*(ft*x*x*x) + ((g2-g1)/ul)*x + g1 + ((ft*ul*ul)/(6*Area*E))*x;
        }
        return 0.;
    };

    cout << "Anaytical Solution" << endl;
    std::vector<double> u_analytical(Nt);
    for(int i = 0 ; i < Nt ; i++){
        u_analytical[i] = u_analytical_x(x[i]);
    }
    for(auto& val : u_analytical){
        cout << std::setprecision(16) << val << endl;
    }

    std::ofstream out("./solutions/solution_2b_iv.dat");
    for (int i = 0; i < x.size(); ++i){
        out << x[i] << " " << u_analytical[i] << " " << D[i] << "\n";
    }

    auto l2_norm_of_error = [ll, ul, he, Nel, quadRule, quad_points, quad_weights, global_x_from_xi, D, basis_function, u_analytical_x](){
        double integral = 0.;
        for(int e = 0 ; e < Nel ; e++){
            for(int i = 0 ; i < quadRule ; i++){
                double u_h = 0.;
                double u_exact = 0.;

                //find u_h at the quad point
                for(int A = 0 ; A < Nne ; A++){
                    u_h += basis_function(A,quad_points[i])*D[e*(Nne - 1) + A];
                }

                //find u_exact at the quad_point
                u_exact = u_analytical_x(global_x_from_xi(e,quad_points[i])); 

                // cout << "calculated at quad point" << quad_points[i] << " " << std::setprecision(16) << u_h << endl;
                // cout << "actual at quad point" << quad_points[i] << " " << std::setprecision(16) << u_exact << endl;

                integral += pow((u_h - u_exact),2)*quad_weights[i];
            }
        }
        return integral*(he/(2*(ul-ll)));
    };

    cout << "l2 norm of error : " << l2_norm_of_error() << endl;

}