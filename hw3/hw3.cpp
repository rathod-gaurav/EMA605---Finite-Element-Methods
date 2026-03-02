//2D time-dependent heat equation

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <Eigen/Dense>
using namespace std;

struct Node{
    double x1; //GLOBAL X1 COORDINATE
    double x2; //GLOBAL X2 COORDINATE
};

template <unsigned int Nne>
struct Element{
    int node[Nne]; //GLOBAL NODE NUMBERS
};

std::tuple<double,double> xi_at_node(unsigned int node){ //function to return xi1 and xi2 for given node A
        double xi1, xi2;
        switch(node){
            case 0:
                xi1 = -1.0;
                xi2 = -1.0;
                break;
            case 1:
                xi1 = 1.0;
                xi2 = -1.0;
                break;
            case 2:
                xi1 = 1.0;
                xi2 = 1.0;
                break;
            case 3:
                xi1 = -1.0;
                xi2 = 1.0;
                break;
            default:
                throw std::invalid_argument("xi_at_node mapping not implemented for this local node number");
        }
        return {xi1, xi2};
};

double basis_function(unsigned int node, double xi1, double xi2){
        auto [xi1_node , xi2_node] = xi_at_node(node);
        double value = 0.25*(1 + xi1*xi1_node)*(1 + xi2*xi2_node);
        return value;
};

std::tuple<double,double> basis_gradient(unsigned int node, double xi1, double xi2){
    auto [xi1_node,xi2_node] = xi_at_node(node);
    double basis_gradient_xi1, basis_gradient_xi2;
    basis_gradient_xi1 = 0.25*xi1_node*(1 + xi2*xi2_node);
    basis_gradient_xi2 = 0.25*xi2_node*(1 + xi1*xi1_node); 
    return {basis_gradient_xi1, basis_gradient_xi2};
}

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

Eigen::MatrixXd extractSubmatrix(const Eigen::MatrixXd& OriginalMatrix , const vector<int> rows , const vector<int> cols){
    Eigen::MatrixXd subMatrix(rows.size(), cols.size());

    for(int i = 0 ; i < rows.size() ; i++){
        for(int j = 0 ; j < cols.size() ; j++){
            subMatrix(i,j) = OriginalMatrix(rows[i],cols[j]);
        }
    }
    return subMatrix;
}

template <unsigned int Nne>
void writeVTK(
    const std::string& filename,
    const std::vector<Node>& nodes,
    const std::vector<Element<Nne>>& elements,
    const Eigen::VectorXd& T
)
{
    static_assert(Nne == 4, "This function supports 4-node quad elements only.");

    ofstream vtkFile(filename + ".vtu");

    if (!vtkFile.is_open()) {
        cerr << "Error opening file.\n";
        return;
    }

    int node_num = nodes.size();
    int element_num = elements.size();

    int vtk_cell_type = 9;   // VTK_QUAD = 9

    // =========================
    // XML Header
    // =========================
    vtkFile << "<?xml version=\"1.0\"?>\n";
    vtkFile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\">\n";
    vtkFile << "<UnstructuredGrid>\n";
    vtkFile << "<Piece NumberOfPoints=\"" << node_num
            << "\" NumberOfCells=\"" << element_num << "\">\n";

    // =========================
    // Write Points
    // =========================
    vtkFile << "<Points>\n";
    vtkFile << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    for (int i = 0; i < node_num; ++i) {
        vtkFile << nodes[i].x1 << " "
                << nodes[i].x2 << " "
                << 0.0 << "\n";  // 2D → z=0
    }

    vtkFile << "</DataArray>\n";
    vtkFile << "</Points>\n";

    // =========================
    // Write Cells
    // =========================
    vtkFile << "<Cells>\n";

    // Connectivity
    vtkFile << "<DataArray type=\"UInt32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int e = 0; e < element_num; ++e) {
        for (int j = 0; j < Nne; ++j) {
            vtkFile << elements[e].node[j] << " ";
        }
        vtkFile << "\n";
    }
    vtkFile << "</DataArray>\n";

    // Offsets
    vtkFile << "<DataArray type=\"UInt32\" Name=\"offsets\" format=\"ascii\">\n";
    int offset = 0;
    for (int e = 0; e < element_num; ++e) {
        offset += Nne;
        vtkFile << offset << "\n";
    }
    vtkFile << "</DataArray>\n";

    // Cell Types
    vtkFile << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int e = 0; e < element_num; ++e) {
        vtkFile << vtk_cell_type << "\n";
    }
    vtkFile << "</DataArray>\n";

    vtkFile << "</Cells>\n";

    // =========================
    // Write Temperature Field
    // =========================
    vtkFile << "<PointData Scalars=\"Temperature\">\n";
    vtkFile << "<DataArray type=\"Float32\" Name=\"Temperature\" format=\"ascii\">\n";

    for (int i = 0; i < node_num; ++i) {
        vtkFile << T(i) << "\n";
    }

    vtkFile << "</DataArray>\n";
    vtkFile << "</PointData>\n";

    // Footer
    vtkFile << "</Piece>\n";
    vtkFile << "</UnstructuredGrid>\n";
    vtkFile << "</VTKFile>\n";

    vtkFile.close();

    cout << "VTK file written successfully: " << filename << ".vtu\n";
}

void initializePVD(const std::string& filename)
{
    std::ofstream pvd(filename);
    pvd << "<?xml version=\"1.0\"?>\n";
    pvd << "<VTKFile type=\"Collection\" version=\"0.1\">\n";
    pvd << "<Collection>\n";
    pvd.close();
}

void appendToPVD(const std::string& filename,
                 const std::string& vtu_filename,
                 double time)
{
    std::ofstream pvd(filename, std::ios::app);

    pvd << "<DataSet timestep=\"" << time
        << "\" group=\"\" part=\"0\" file=\""
        << vtu_filename << "\"/>\n";

    pvd.close();
}

void finalizePVD(const std::string& filename)
{
    std::ofstream pvd(filename, std::ios::app);
    pvd << "</Collection>\n";
    pvd << "</VTKFile>\n";
    pvd.close();
}

int main(){
    unsigned int Nd = 2; //2D problem
    constexpr int Nne = 4; //number of nodes in an element //3 for triangular elements, 4 for quadrilateral elements
    unsigned int bfnOrder = 1; //basis function order
    unsigned int quadRule = 2; //quardature rule

    //problem variables
    double conductivity = 1.0;// 385.0;
    double rho_c = 1.0; //3.8151e6;
    double fint = 0.0; //internal forcing function
    double h = 0.0; //neumann boundary condition

    //domain
    double x1_ll = 0.; //lower limit of x1 dimension
    double x1_ul = 1.; //upper limit of x1 dimension
    double x2_ll = 0.; //lower limit of x2 dimension
    double x2_ul = 1.; //upper limit of x2 dimension

    //create mesh
    unsigned int Nel_x1 = 20; //number of elements in x1 direction
    unsigned int Nel_x2 = 20; //number of elements in x2 direction
    
    unsigned int Nnodes_x1 = Nel_x1 + 1; //number of nodes in x1 direction
    unsigned int Nnodes_x2 = Nel_x2 + 1; //number of nodes in x2 direction

    double dx1 = (x1_ul - x1_ll)/Nel_x1;
    double dx2 = (x2_ul - x2_ll)/Nel_x2;

    unsigned int Nel_t = Nel_x1*Nel_x2; //total number of elements

    unsigned int Nt = Nnodes_x1*Nnodes_x2; //total number of nodes

    //global node locations
    vector<Node> nodes;
    nodes.reserve(Nt);
    for(int j = 0 ; j < Nnodes_x2 ; j++){
        for(int i = 0 ; i < Nnodes_x1 ; i++){
            Node n;
            n.x1 = x1_ll + i*dx1;
            n.x2 = x2_ll + j*dx2;
            nodes.push_back(n);
        }
    }

    // cout << nodes.size() << endl;

    using Element2D = Element<Nne>;
    vector<Element2D> elements;
    elements.reserve(Nel_t);

    for(int j = 0 ; j < Nel_x2 ; j++){
        for(int i = 0 ; i < Nel_x1 ; i++){
            Element2D elem;
            int n0 = i + j*Nnodes_x1;
            int n1 = n0 + 1;
            int n2 = Nnodes_x1 + i + j*Nnodes_x1 + 1;
            int n3 = n2 - 1;

            elem.node[0] = n0;
            elem.node[1] = n1;
            elem.node[2] = n2;
            elem.node[3] = n3;

            elements.push_back(elem);
        }
    }
    // cout << elements.size() << endl;

    std::ofstream points_file("points.txt");
    for(auto& node : nodes){
        points_file << node.x1 << " " << node.x2 << "\n";
    }

    std::ofstream quads_file("quads.txt");
    for(auto& elem : elements){
        quads_file << elem.node[0] << " " << elem.node[1] << " " << elem.node[2] << " " << elem.node[3] << "\n";
    }

    auto calculate_Jacobian = [Nd, elements, nodes](int e, double xi1, double xi2){//function to calculate jacobian
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(Nd,Nd);
        
        for(int A = 0 ; A < Nne ; A++){
            auto [basis_gradient_xi1, basis_gradient_xi2] = basis_gradient(A, xi1, xi2);
            unsigned int Aglobal = elements[e].node[A];
            J(0,0) += basis_gradient_xi1*nodes[Aglobal].x1; //dx1/dxi1
            J(0,1) += basis_gradient_xi2*nodes[Aglobal].x1; //dx1/dxi2
            J(1,0) += basis_gradient_xi1*nodes[Aglobal].x2; //dx2/dxi1
            J(1,1) += basis_gradient_xi2*nodes[Aglobal].x2; //dx2/dxi2
        }
        return J;
    };

    auto calculate_Kappa = [conductivity,Nd](){//function to return the element conductivity
        Eigen::MatrixXd Kappa = Eigen::MatrixXd::Zero(Nd,Nd);
        for(int i = 0 ; i < Nd ; i++){
            for(int j = 0 ; j < Nd ; j++){
                if(i==j){
                    Kappa(i,j) = conductivity;
                }
                else{
                    Kappa(i,j) = 0.0;
                }       
            }
        }
        return Kappa;
    };

    auto global_x_from_xi = [Nd, Nne, elements, nodes](int e, double xi1, double xi2){//function to calculate global x coordinates for given quadrature points
        Eigen::VectorXd xglobal = Eigen::VectorXd::Zero(Nd);
        for(int A = 0 ; A < Nne ; A++){
            int Aglobal = elements[e].node[A];
            xglobal(0) += basis_function(A, xi1, xi2)*nodes[Aglobal].x1;
            xglobal(1) += basis_function(A, xi1, xi2)*nodes[Aglobal].x2;
        }
        return xglobal;
    };
    
    //Quadrature points
    QuadratureRule q = gauss_legendre(quadRule);
    std::vector<double> points(quadRule), weights(quadRule);
    points = q.points;
    weights = q.weights;
    Eigen::VectorXd quad_points = Eigen::Map<Eigen::VectorXd>(points.data(), points.size());
    Eigen::VectorXd quad_weights = Eigen::Map<Eigen::VectorXd>(weights.data(), weights.size());

    //Global node locations having Neumann Boundary conditions specified on them
    vector<int> nodeLocationsN;
    for(int i = 0; i < Nt; i++){
        if(nodes[i].x2 == x2_ll || nodes[i].x2 == x2_ul){
            nodeLocationsN.push_back(i);
        }
    }
    vector<bool> isNeumann(Nt,false);
    for(int& nodeLocation : nodeLocationsN){
        isNeumann[nodeLocation] = true;
    }
    double he = nodes[1].x1 - nodes[0].x1;//node spacing in x1 direction : to be used for computation of neumann boundary condition term

    //Assembly
    Eigen::MatrixXd Kglobal = Eigen::MatrixXd::Zero(Nt,Nt);
    Eigen::VectorXd Fglobal = Eigen::VectorXd::Zero(Nt);
    Eigen::MatrixXd Mglobal = Eigen::MatrixXd::Zero(Nt,Nt);

    for(int e = 0; e < Nel_t; e++){
        Eigen::MatrixXd Klocal = Eigen::MatrixXd::Zero(Nne,Nne);
        Eigen::MatrixXd Mlocal = Eigen::MatrixXd::Zero(Nne,Nne);
        Eigen::VectorXd Flocal_int = Eigen::VectorXd::Zero(Nne);
        Eigen::VectorXd Flocal_h = Eigen::VectorXd::Zero(Nne);
        for(int A = 0; A < Nne ; A++){
            for(int B = 0 ; B < Nne ; B++){
                for(int I = 0; I < quadRule ; I++){
                    for(int J = 0; J < quadRule ; J++){
                        double xi1 = quad_points(I);
                        double xi2 = quad_points(J); 
                        // Eigen::VectorXd x = global_x_from_xi(e, xi1, xi2);
                        // double x1 = x(0);
                        // double x2 = x(1);

                        auto Jac = calculate_Jacobian(e,xi1,xi2);
                        auto Jac_inv = Jac.inverse();

                        auto [bfgradientA_xi1 , bfgradientA_xi2] = basis_gradient(A,xi1,xi2);
                        auto [bfgradientB_xi1 , bfgradientB_xi2] = basis_gradient(B,xi1,xi2);

                        Eigen::VectorXd bfgradientA = Eigen::VectorXd::Zero(Nd);
                        Eigen::VectorXd bfgradientB = Eigen::VectorXd::Zero(Nd);
                        bfgradientA << bfgradientA_xi1 , bfgradientA_xi2;
                        bfgradientB << bfgradientB_xi1 , bfgradientB_xi2;

                        Eigen::MatrixXd Kappa = calculate_Kappa();

                        Eigen::MatrixXd Kvalue = (bfgradientA.transpose()*Jac_inv)*Kappa*(Jac_inv.transpose()*bfgradientB);
                        Klocal(A,B) += Kvalue(0,0)*quad_weights(I)*quad_weights(J)*Jac.determinant();

                        Mlocal(A,B) += basis_function(A,xi1,xi2)*rho_c*basis_function(B,xi1,xi2)*quad_weights(I)*quad_weights(J)*Jac.determinant();
                    }
                }
            }
            for(int I = 0 ; I < quadRule ; I++){
                for(int J = 0 ; J < quadRule ; J++){
                    double xi1 = quad_points(I);
                    double xi2 = quad_points(J); 

                    auto Jac = calculate_Jacobian(e,xi1,xi2);
                    Flocal_int(A) += basis_function(A,xi1,xi2)*fint*quad_weights(I)*quad_weights(J)*Jac.determinant();
                }
            }
            int globalNode = elements[e].node[A];
            if(isNeumann[globalNode]){
                double xi2 = 0.0;
                if(nodes[globalNode].x2 == x2_ul){
                    xi2 = 1.0;
                }
                else if(nodes[globalNode].x2 == x2_ll){
                    xi2 = -1.0;
                }
                for(int I = 0 ; I < quadRule ; I++){
                    double xi1 = quad_points(I);
                    double Jac = he/2;
                    Flocal_h(A) += basis_function(A,xi1,xi2)*h*Jac*quad_weights(I); 
                }
            }
            else{
                Flocal_h(A) = 0;
            }
        }
        //Assembly
        // cout << Klocal << endl;
        // cout << endl;
        for(int A = 0 ; A < Nne ; A++){
            int Aglobal = elements[e].node[A];
            for(int B = 0 ; B < Nne ; B++){
                int Bglobal = elements[e].node[B];
                Kglobal(Aglobal,Bglobal) += Klocal(A,B);
                Mglobal(Aglobal,Bglobal) += Mlocal(A,B);
            }
            Fglobal(Aglobal) += Flocal_int(A) - Flocal_h(A); //this now contains contribution from neumann boundary condition as well
        }
    }

    // std::ofstream Kglobal_file("Kglobal.txt");
    // Kglobal_file << Kglobal;

    // std::ofstream Mglobal_file("Mglobal.txt");
    // Mglobal_file << Mglobal;

    // std::ofstream Fglobal_file("Fglobal.txt");
    // Fglobal_file << Fglobal;

    //global nodesLocations having dirichlet boundary condition specified on them
    vector<int> nodeLocationsD; //knowns (dirichlet node locations)
    for(int i = 0 ; i < Nt ; i++){
        if(nodes[i].x1== x1_ll || nodes[i].x1 == x1_ul){ //dirichlet boundary at x = 0 and x = 1
            nodeLocationsD.push_back(i);
        }
    }
    // std::ofstream LocationsD("nodeLocationsD.txt");
    // for(auto& nodeLocation : nodeLocationsD){
    //     LocationsD << nodeLocation << "\n";
    // }

    //values of dirichlet boundary
    Eigen::VectorXd dirischletVal(nodeLocationsD.size());
    for(int i = 0 ; i < nodeLocationsD.size() ; i++){
        int nodeD = nodeLocationsD[i];
        if(nodes[nodeD].x1 == x1_ll){
            dirischletVal[i] = 300.0;
        }
        else if(nodes[nodeD].x1 == x1_ul){
            dirischletVal[i] = 310.0;
        }
    }
    Eigen::VectorXd dirischletValDot = Eigen::VectorXd::Zero(nodeLocationsD.size()); //rate of change of temperature on the dirischlet boundary
    // std::ofstream ValD("DirischletVal.txt");
    // for(auto& Val : dirischletVal){
    //     ValD << Val << "\n";
    // }

    //Applying Dirischlet Boundary Conditions
    vector<int> nodeLocationsU; //unknown node locations - node locations where fleid value is unknown
    vector<bool> isDirischlet(Nt,false);
    for(int& nodeLocation : nodeLocationsD){
        isDirischlet[nodeLocation] = true;
    }
    for(int i = 0 ; i < Nt ; i++){
        if(!isDirischlet[i]){
            nodeLocationsU.push_back(i);
        }
    }
    // std::ofstream LocationsU("nodeLocationsU.txt");
    // for(auto& nodeLocation : nodeLocationsU){
    //     LocationsU << nodeLocation << "\n";
    // }

        
    Eigen::MatrixXd KUU = extractSubmatrix(Kglobal, nodeLocationsU, nodeLocationsU); //extract from Kglobal - only rows and columns pertaining to unknown node locations
    Eigen::MatrixXd KUD = extractSubmatrix(Kglobal, nodeLocationsU, nodeLocationsD); //extract from Kglobal - only columns corresponding to Dirischlet node locations, for rows corresponding to unknown node locations

    Eigen::MatrixXd MUU = extractSubmatrix(Mglobal, nodeLocationsU, nodeLocationsU); //extract from Mglobal - only rows and columns pertaining to unknown node locations
    Eigen::MatrixXd MUD = extractSubmatrix(Mglobal, nodeLocationsU, nodeLocationsD); //extract from Mglobal - only columns corresponding to Dirischlet node locations, for rows corresponding to unknown node locations

    Eigen::VectorXd FU(nodeLocationsU.size()); //extract from Fglobal - only rows corresponding to unknown node locations
    for(int i = 0; i < nodeLocationsU.size(); i++){
        FU(i) = Fglobal(nodeLocationsU[i]);
    }
    
    Eigen::VectorXd F(FU.size()); //create final forcing function vector
    F = FU - KUD*dirischletVal - MUD*dirischletValDot;
    
    //Time integration
    double alpha = 1.0;
    double dt = 0.01; //time step size
    int NT = 100; //number of time steps
    // Initial condition
    Eigen::VectorXd D0 = Eigen::VectorXd::Zero(Nt);
    Eigen::VectorXd V0 = Eigen::VectorXd::Zero(Nt);
    for(int i = 0 ; i < Nt ; i++){
        if(nodes[i].x1 < 0.5){
            D0(i) = 300.0;
        }
        else if(nodes[i].x1 >= 0.5){
            D0(i) = 300.0 + 20*(nodes[i].x1 - 0.5);
        }
    }
    
    Eigen::VectorXd Dn(nodeLocationsU.size());
    Eigen::VectorXd Vn(nodeLocationsU.size());
    for(int i = 0; i < nodeLocationsU.size() ; i++){
        Dn(i) = D0(nodeLocationsU[i]);
    }
    Eigen::LDLT<Eigen::MatrixXd> solver1(MUU);
    Vn = solver1.solve(F - KUU*Dn); //find V0 to initiate the time stepping process
    // cout << Vn << endl;
    Eigen::MatrixXd lhs = MUU + alpha*dt*KUU;
    Eigen::LDLT<Eigen::MatrixXd> solver(lhs);

    //Final solution stored in D
    Eigen::VectorXd D = Eigen::VectorXd::Zero(Nt);

    initializePVD("solutions/heat.pvd");

    // Eigen::VectorXd Dnp1 = Eigen::VectorXd::Zero(Dn.size());

    std::string filename = "solutions/solution_0.txt";
    ofstream D_file(filename);
    D_file << D0;

    for(int n = 0 ; n < NT ; n++){
        Eigen::VectorXd predictor = Dn + dt*(1-alpha)*Vn;
        Eigen::VectorXd rhs = alpha*dt*F + MUU*predictor;
        
        Eigen::VectorXd Dnp1 = solver.solve(rhs);

        Dn = Dnp1;
        Vn = (Dnp1 - predictor)/(alpha*dt);

         // apply boundary conditions to obtain final solution
        for(int i = 0 ; i < nodeLocationsD.size() ; i++){
            int indexD = nodeLocationsD[i];
            D[indexD] = dirischletVal[i];
        }
        for(int i = 0 ; i < nodeLocationsU.size() ; i++){
            int indexD = nodeLocationsU[i];
            D[indexD] = Dn[i];
        }

        // cout << D.size() << endl;
        std::string basename = "heat_" + std::to_string(n);
        std::string filename = "solutions/" + basename;
        std::string vtu_name = basename + ".vtu";
        writeVTK<Nne>(filename, nodes, elements, D);
        appendToPVD("solutions/heat.pvd", vtu_name, n*dt);
        // if(n==9 || n == 24 || n == 99)
        // {
        //     std::string filename = "solutions/solution_" + std::to_string(n) + ".txt";
        //     ofstream D_file(filename);
        //     D_file << D;
        // }

    }    
    
    finalizePVD("solutions/heat.pvd");

    

    // debugging block start
    // int Aglobal = 88;
    // cout << nodes[Aglobal].x1 << " " << nodes[Aglobal].x2 << endl;
    // int e = 0;
    // int Alocal = 2;
    // cout << elements[e].node[Alocal] << endl;

    // auto [xi1,xi2] = xi_at_node(5);
    // cout << xi1 << " " << xi2 << endl;
    // cout << basis_function(1, 1,-1) << endl;
    // auto [basis_gradient_xi1, basis_gradient_xi2] = basis_gradient(0,-1,-1);
    // cout << basis_gradient_xi1 << " " << basis_gradient_xi2 << endl;

    // auto J = Jacobian(100, 0.5, 0.5);
    // cout << J << endl;
    // cout << J.inverse() << endl;

    // auto xglobal = global_x_from_xi(1,-1,-1);
    // cout << xglobal << endl;

    // cout << quad_points << endl;
    // cout << quad_weights << endl;

    // debugging block end
}