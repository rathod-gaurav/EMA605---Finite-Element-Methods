// Linear Hyperbolic PDE with vector variable - 3D elasto dynamics
// no body forces, no neumann conditions, only dirischlet boundary conditions at top and bottom faces of the domain

#include <iostream>
using namespace std;
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <map>

struct Node{
    float x1, x2, x3;
};

template <unsigned int Nne>
struct Element{
    int node[Nne];
};

std::tuple<float,float,float> xi_at_node(unsigned int node){ //function to return xi1 and xi2 for given node A
        float xi1, xi2, xi3;
        switch(node){
            case 0:
                xi1 = -1.0;
                xi2 = -1.0;
                xi3 = -1.0;
                break;
            case 1:
                xi1 = 1.0;
                xi2 = -1.0;
                xi3 = -1.0;
                break;
            case 2:
                xi1 = 1.0;
                xi2 = 1.0;
                xi3 = -1.0;
                break;
            case 3:
                xi1 = -1.0;
                xi2 = 1.0;
                xi3 = -1.0;
                break;
            case 4:
                xi1 = -1.0;
                xi2 = -1.0;
                xi3 = 1.0;
                break;
            case 5:
                xi1 = 1.0;
                xi2 = -1.0;
                xi3 = 1.0;
                break;
            case 6:
                xi1 = 1.0;
                xi2 = 1.0;
                xi3 = 1.0;
                break;
            case 7:
                xi1 = -1.0;
                xi2 = 1.0;
                xi3 = 1.0;
                break;
            default:
                throw std::invalid_argument("xi_at_node mapping not implemented for this local node number");
        }
        return {xi1, xi2, xi3};
};

float basis_function(unsigned int node, float xi1, float xi2, float xi3){
        auto [xi1_node , xi2_node , xi3_node] = xi_at_node(node);
        float value = 0.125*(1 + xi1*xi1_node)*(1 + xi2*xi2_node)*(1 + xi3*xi3_node);
        return value;
};

std::tuple<float,float,float> basis_gradient(unsigned int node, float xi1, float xi2, float xi3){
    auto [xi1_node,xi2_node,xi3_node] = xi_at_node(node);
    float basis_gradient_xi1, basis_gradient_xi2, basis_gradient_xi3;
    basis_gradient_xi1 = 0.125*xi1_node*(1 + xi2*xi2_node)*(1 + xi3*xi3_node);
    basis_gradient_xi2 = 0.125*xi2_node*(1 + xi1*xi1_node)*(1 + xi3*xi3_node);
    basis_gradient_xi3 = 0.125*xi3_node*(1 + xi1*xi1_node)*(1 + xi2*xi2_node);
    return {basis_gradient_xi1, basis_gradient_xi2, basis_gradient_xi3};
}

struct QuadratureRule {
    std::vector<float> points;
    std::vector<float> weights;
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

Eigen::MatrixXf extractSubmatrix(const Eigen::MatrixXf& OriginalMatrix , const vector<int> rows , const vector<int> cols){//extract submatrix from original matrix given row and column indexes
    Eigen::MatrixXf subMatrix(rows.size(), cols.size());

    for(int i = 0 ; i < rows.size() ; i++){
        for(int j = 0 ; j < cols.size() ; j++){
            subMatrix(i,j) = OriginalMatrix(rows[i],cols[j]);
        }
    }
    return subMatrix;
}

template <unsigned int Nne>
void write_vtu(
    const std::string& filename,
    const std::vector<Node>& nodes,
    const std::vector<Element<Nne>>& elements,
    const Eigen::VectorXf& displacement   // size = 3*Nnodes
)
{
    std::ofstream file(filename);

    int Nnodes = nodes.size();
    int Nelems = elements.size();

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "<UnstructuredGrid>\n";
    file << "<Piece NumberOfPoints=\"" << Nnodes 
         << "\" NumberOfCells=\"" << Nelems << "\">\n";

    // ---------------------
    // POINTS
    // ---------------------
    file << "<Points>\n";
    file << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    for (int i = 0; i < Nnodes; ++i)
        file << nodes[i].x1 << " "
             << nodes[i].x2 << " "
             << nodes[i].x3 << "\n";

    file << "</DataArray>\n</Points>\n";

    // ---------------------
    // CELLS (Hex = VTK type 12)
    // ---------------------
    file << "<Cells>\n";

    // connectivity
    file << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int e = 0; e < Nelems; ++e)
        for (int A = 0; A < 8; ++A)
            file << elements[e].node[A] << " ";
    file << "\n</DataArray>\n";

    // offsets
    file << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int e = 0; e < Nelems; ++e)
        file << (e+1)*8 << " ";
    file << "\n</DataArray>\n";

    // types (12 = hexahedron)
    file << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int e = 0; e < Nelems; ++e)
        file << 12 << " ";
    file << "\n</DataArray>\n";

    file << "</Cells>\n";

    // ---------------------
    // POINT DATA (Displacement)
    // ---------------------
    file << "<PointData Vectors=\"Displacement\">\n";
    file << "<DataArray type=\"Float32\" Name=\"Displacement\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    for (int i = 0; i < Nnodes; ++i)
    {
        file << displacement(3*i)   << " "
             << displacement(3*i+1) << " "
             << displacement(3*i+2) << "\n";
    }

    file << "</DataArray>\n</PointData>\n";

    file << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
}

void write_pvd(
    const std::string& filename,
    const std::vector<std::string>& vtu_files,
    const std::vector<float>& times)
{
    std::ofstream file(filename);

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "<Collection>\n";

    for (size_t i = 0; i < vtu_files.size(); ++i)
    {
        file << "<DataSet timestep=\"" << times[i]
             << "\" group=\"\" part=\"0\" file=\""
             << vtu_files[i] << "\"/>\n";
    }

    file << "</Collection>\n</VTKFile>\n";
}

int main(){
    unsigned int Nsd = 3; //number of spatial dimensions - 3D problem
    constexpr int Nne = 8; //number of nodes in an element - 8 for hexahedral element
    unsigned int quadRule = 2; //quadrature rule for numerical integration

    //problem variables
    float E = 1e3;
    float nu = 0.3;
    float mu = (E*nu)/((1 + nu)*(1 - 2*nu));
    float lambda = E/(2*(1 + nu));
    float rho = 1.0;

    //Rayleigh Damping Constants
    float a = 1.0;
    float b = 0.001;

    //domain
    float x1_ll = 0.0;
    float x1_ul = 0.1;
    float x2_ll = 0.0;
    float x2_ul = 0.1;
    float x3_ll = 0.0;
    float x3_ul = 1.0;

    //Mesh
    unsigned int Nel_x1 = 4; //number of elements in x1 direction
    unsigned int Nel_x2 = 4; //number of elements in x2 direction
    unsigned int Nel_x3 = 40; //number of elements in x3 direction

    unsigned int Nnodes_x1 = Nel_x1 + 1; //number of nodes in x1 direction
    unsigned int Nnodes_x2 = Nel_x2 + 1; //number of nodes in x2 direction
    unsigned int Nnodes_x3 = Nel_x3 + 1; //number of nodes in x3 direction

    float dx1 = (x1_ul - x1_ll)/Nel_x1;
    float dx2 = (x2_ul - x2_ll)/Nel_x2;
    float dx3 = (x3_ul - x3_ll)/Nel_x3;

    unsigned int Nel_t = Nel_x1*Nel_x2*Nel_x3; //total number of elements in the domain
    unsigned int Nt = Nnodes_x1*Nnodes_x2*Nnodes_x3; //total number of nodes in the domain

    //Global node locations
    vector<Node> nodes;
    nodes.reserve(Nt);
    for(unsigned int k = 0 ; k < Nnodes_x3 ; k++){
        for(unsigned int j = 0 ; j < Nnodes_x2 ; j++){
            for(unsigned int i = 0 ; i < Nnodes_x1 ; i++){
                Node n;
                n.x1 = x1_ll + i*dx1;
                n.x2 = x2_ll + j*dx2;
                n.x3 = x3_ll + k*dx3;
                nodes.push_back(n);
            }
        }
    }

    //Local-Global node number mapping for every element
    using Element3D = Element<Nne>;
    vector<Element3D> elements;
    elements.reserve(Nel_t);
    for(unsigned int k = 0 ; k < Nel_x3 ; k++){
        for(unsigned int j = 0 ; j < Nel_x2 ; j++){
            for(unsigned int i = 0 ; i < Nel_x1 ; i++){
                Element3D elem;
                // int n0 = i + j*Nnodes_x1 + k*(Nnodes_x1*Nnodes_x2);
                // int n1 = n0 + 1;
                // int n2 = n1 + (Nnodes_x1*Nnodes_x2);
                // int n3 = n2 - 1;
                // int n4 = i + (j+1)*Nnodes_x1 + k*(Nnodes_x1*Nnodes_x2);
                // int n5 = n4 + 1;
                // int n6 = n5 + (Nnodes_x1*Nnodes_x2);
                // int n7 = n6 - 1;

                int base = i 
                     + j * Nnodes_x1 
                     + k * (Nnodes_x1 * Nnodes_x2);

                int n0 = base;
                int n1 = base + 1;
                int n3 = base + Nnodes_x1;
                int n2 = n3 + 1;

                int n4 = base + Nnodes_x1 * Nnodes_x2;
                int n5 = n4 + 1;
                int n7 = n4 + Nnodes_x1;
                int n6 = n7 + 1;

                elem.node[0] = n0;
                elem.node[1] = n1;
                elem.node[2] = n2;
                elem.node[3] = n3;
                elem.node[4] = n4;
                elem.node[5] = n5;
                elem.node[6] = n6;
                elem.node[7] = n7;

                elements.push_back(elem);
            }
        }
    }

    //store the mesh into points and hexa files
    std::ofstream points_file("points.txt");
    for(auto& node : nodes){
        points_file << node.x1 << " " << node.x2  << " " << node.x3 << "\n";
    }

    std::ofstream hexas_file("hexas.txt");
    for(auto& elem : elements){
        hexas_file << elem.node[0] << " " << elem.node[1] << " " << elem.node[2] << " " << elem.node[3] << " " << elem.node[4] << " " << elem.node[5] << " " << elem.node[6] << " " << elem.node[7] << "\n";
    }

    //Isoparametric Mapping - Calculate 3D jacobian
    auto calculate_Jacobian_3D = [Nsd, elements, nodes](int e, float xi1, float xi2, float xi3){//function to calculate jacobian
        Eigen::MatrixXf J = Eigen::MatrixXf::Zero(Nsd,Nsd);
        
        for(int A = 0 ; A < Nne ; A++){
            auto [basis_gradient_xi1, basis_gradient_xi2, basis_gradient_xi3] = basis_gradient(A, xi1, xi2, xi3);
            int Aglobal = elements[e].node[A];
            J(0,0) += basis_gradient_xi1*nodes[Aglobal].x1; //dx1/dxi1
            J(0,1) += basis_gradient_xi2*nodes[Aglobal].x1; //dx1/dxi2
            J(0,2) += basis_gradient_xi3*nodes[Aglobal].x1; //dx1/dxi3
            J(1,0) += basis_gradient_xi1*nodes[Aglobal].x2; //dx2/dxi1
            J(1,1) += basis_gradient_xi2*nodes[Aglobal].x2; //dx2/dxi2
            J(1,2) += basis_gradient_xi3*nodes[Aglobal].x2; //dx2/dxi3
            J(2,0) += basis_gradient_xi1*nodes[Aglobal].x3; //dx3/dxi1
            J(2,1) += basis_gradient_xi2*nodes[Aglobal].x3; //dx3/dxi2
            J(2,2) += basis_gradient_xi3*nodes[Aglobal].x3; //dx3/dxi3
        }
        return J;
    };

    //get global location corresponding to xi1, xi2, xi3 of any point in the parent domain
    auto global_x_from_xi = [Nsd, Nne, elements, nodes](int e, float xi1, float xi2, float xi3){//function to calculate global x coordinates for given quadrature points
        Eigen::VectorXf xglobal = Eigen::VectorXf::Zero(Nsd);
        for(int A = 0 ; A < Nne ; A++){
            int Aglobal = elements[e].node[A];
            xglobal(0) += basis_function(A, xi1, xi2, xi3)*nodes[Aglobal].x1;
            xglobal(1) += basis_function(A, xi1, xi2, xi3)*nodes[Aglobal].x2;
            xglobal(2) += basis_function(A, xi1, xi2, xi3)*nodes[Aglobal].x3;
        }
        return xglobal;
    };

    //Quadrature points
    QuadratureRule q = gauss_legendre(quadRule);
    std::vector<float> points(quadRule), weights(quadRule);
    points = q.points;
    weights = q.weights;
    Eigen::VectorXf quad_points = Eigen::Map<Eigen::VectorXf>(points.data(), points.size());
    Eigen::VectorXf quad_weights = Eigen::Map<Eigen::VectorXf>(weights.data(), weights.size());

    Eigen::MatrixXf Kglobal = Eigen::MatrixXf::Zero(Nt*Nsd,Nt*Nsd);
    Eigen::VectorXf Fglobal = Eigen::VectorXf::Zero(Nt*Nsd);
    Eigen::MatrixXf Mglobal = Eigen::MatrixXf::Zero(Nt*Nsd,Nt*Nsd);

    for(int e = 0; e < Nel_t ; e++){
        Eigen::MatrixXf Klocal = Eigen::MatrixXf::Zero(Nne*Nsd,Nne*Nsd);
        Eigen::MatrixXf Mlocal = Eigen::MatrixXf::Zero(Nne*Nsd,Nne*Nsd);
        //Construct local matrices for element e
        for(int A = 0 ; A < Nne ; A++){
            for(int B = 0 ; B < Nne ; B++){
                for(int I = 0 ; I < quadRule ; I++){
                    for(int J = 0 ; J < quadRule ; J++){
                        for(int K = 0 ; K < quadRule ; K++){
                            float xi1 = quad_points(I);
                            float xi2 = quad_points(J);
                            float xi3 = quad_points(K);

                            Eigen::MatrixXf Jac = calculate_Jacobian_3D(e, xi1, xi2, xi3);
                            if(Jac.determinant() < 0){ 
                                throw std::runtime_error("Negative Jacobian detected");
                                break;
                            }
                            Eigen::MatrixXf Jac_inv = Jac.inverse();

                            auto [bfgradientA_xi1 , bfgradientA_xi2 , bfgradientA_xi3] = basis_gradient(A, xi1, xi2, xi3);
                            auto [bfgradientB_xi1 , bfgradientB_xi2 , bfgradientB_xi3] = basis_gradient(B, xi1, xi2, xi3);

                            Eigen::VectorXf bfgradientA = Eigen::VectorXf::Zero(Nsd);
                            Eigen::VectorXf bfgradientB = Eigen::VectorXf::Zero(Nsd);

                            bfgradientA << bfgradientA_xi1 , bfgradientA_xi2 , bfgradientA_xi3;
                            bfgradientB << bfgradientB_xi1 , bfgradientB_xi2 , bfgradientB_xi3;

                            Eigen::VectorXf term1 = Jac_inv.transpose()*bfgradientA;
                            Eigen::VectorXf term2 = Jac_inv.transpose()*bfgradientB;
                            Eigen::MatrixXf II = Eigen::MatrixXf::Identity(Nsd,Nsd);

                            Eigen::MatrixXf Klocal_mu = 2*mu*(term1.dot(term2))*II*Jac.determinant()*quad_weights(I)*quad_weights(J)*quad_weights(K);
                            Eigen::MatrixXf Klocal_lambda = lambda*(term1*term2.transpose())*Jac.determinant()*quad_weights(I)*quad_weights(J)*quad_weights(K);

                            Eigen::MatrixXf Kblock = Klocal_mu + Klocal_lambda;
                            Eigen::MatrixXf Mblock = rho*basis_function(A, xi1, xi2, xi3)*basis_function(B, xi1, xi2, xi3)*Jac.determinant()*quad_weights(I)*quad_weights(J)*quad_weights(K)*II;

                            Klocal.block<3,3>(3*A, 3*B) += Kblock;
                            Mlocal.block<3,3>(3*A, 3*B) += Mblock;
                        }
                    }
                }
            }
        }

        //Assemble
        for(int A = 0; A < Nne; A++){
            int Aglobal = elements[e].node[A];
            for(int B = 0; B < Nne ; B++)
            {
                int Bglobal = elements[e].node[B];
                Kglobal.block<3,3>(3*Aglobal,3*Bglobal) += Klocal.block<3,3>(3*A,3*B);
                Mglobal.block<3,3>(3*Aglobal,3*Bglobal) += Mlocal.block<3,3>(3*A,3*B);
            }
        }
        // cout << "Assembled into Kglobal for element : " << e << endl;
    }

    // std::ofstream Kglobal_file("Kglobal.txt");
    // Kglobal_file << Kglobal;
    
    //Boundary Conditions
    //global nodelocations where dirischlet boundary conditions are specified
    std::map<int, std::vector<int>> nodeLocationsD_map;
    for(int i = 0 ; i < Nt ; i++){
        if(nodes[i].x3 == x3_ll){
            nodeLocationsD_map[i].push_back(0); // 0 => X1 displacement specified on this node 
            nodeLocationsD_map[i].push_back(1); // 1 => X2 displacement specified on this node
            nodeLocationsD_map[i].push_back(2); // 2 => X3 displacement specified on this node
        }
        else if(nodes[i].x3 == x3_ul){
            nodeLocationsD_map[i].push_back(1); // 1=> only X2 displacement specified on this node
        }
    }
    vector<bool> isDirischlet(Nt,false);
    for(const auto& [key,vec] : nodeLocationsD_map){
        isDirischlet[key] = true;
    }
    //print this DirischletMap
    // for (const auto& [key, vec] : nodeLocationsD_map) {
    //     std::cout << key << " -> [ ";
    //     for (int v : vec) std::cout << v << " ";
    //     std::cout << "]\n";
    // }
    

    //indexes to remove from the solution array corresponding to dirischlet boundary conditions
    vector<int> dirischletIndexes;
    for(int i = 0 ; i < Nt ; i++){
        if(isDirischlet[i]){
            for(int dof : nodeLocationsD_map[i]){
                dirischletIndexes.push_back(3*i + dof);
            }
        }
    }
    vector<int> unknownIndexes;
    for(int i = 0 ; i < Nt ; i++){
        if(!isDirischlet[i]){
            unknownIndexes.push_back(3*i);
            unknownIndexes.push_back(3*i + 1);
            unknownIndexes.push_back(3*i + 2);
        }
    }


    //given values of displacement field at dirischlet boundary
    Eigen::VectorXf dirischletVal(dirischletIndexes.size());
    for(int i = 0 ; i < dirischletIndexes.size() ; i++){
        int nodeD = dirischletIndexes[i]/3;
        int dof = dirischletIndexes[i]%3;
        if(nodes[nodeD].x3 == x3_ll){
            dirischletVal(i) = 0.0; //all displacements are 0 at bottom face
        }
        else if(nodes[nodeD].x3 == x3_ul){
            if(dof == 1){ //only x2 displacement is specified at top face
                dirischletVal(i) = 0.05; 
            }
        }
    } 

    Eigen::VectorXf dirischletValDot = Eigen::VectorXf::Zero(dirischletIndexes.size()); //rate of change of displacement at dirischlet boundary - 0 for this problem since we are applying static dirischlet boundary conditions

    Eigen::MatrixXf KUU = extractSubmatrix(Kglobal, unknownIndexes, unknownIndexes); //extract from Kglobal - only rows and columns pertaining to unknown node locations
    Eigen::MatrixXf KUD = extractSubmatrix(Kglobal, unknownIndexes, dirischletIndexes); //extract from Kglobal - only columns corresponding to Dirischlet node locations, for rows corresponding to unknown node locations
    
    Eigen::MatrixXf MUU = extractSubmatrix(Mglobal, unknownIndexes, unknownIndexes); //extract from Mglobal - only rows and columns pertaining to unknown node locations
    Eigen::MatrixXf MUD = extractSubmatrix(Mglobal, unknownIndexes, dirischletIndexes); //extract from Mglobal - only columns corresponding to Dirischlet node locations, for rows corresponding to unknown node locations

    Eigen::VectorXf FU(unknownIndexes.size()); //extract from Fglobal - only rows corresponding to unknown node locations
    for(int i = 0; i < unknownIndexes.size(); i++){
        FU(i) = Fglobal(unknownIndexes[i]);
    }

    Eigen::VectorXf F(FU.size()); //final forcing function vector after applying dirischlet boundary conditions
    F = FU - KUD*dirischletVal - MUD*dirischletValDot;

    Eigen::MatrixXf CUU = a*MUU + b*KUU; //Rayleigh Damping Matrix

    //Initial conditions - initial displacement and velocity are 0 everywhere in the domain
    Eigen::VectorXf D0_full = Eigen::VectorXf::Zero(Nt*Nsd);
    Eigen::VectorXf D0 = Eigen::VectorXf::Zero(unknownIndexes.size()); //initial displacement at unknown node locations
    Eigen::VectorXf Ddot0 = Eigen::VectorXf::Zero(unknownIndexes.size()); //initial velocity at unknown node locations
    Eigen::VectorXf Ddotdot0 = MUU.inverse()*(F - CUU*Ddot0 - KUU*D0); //initial acceleration at unknown node locations

    //Final solution vector to be constructed after time integration, including known values at dirischlet boundary
    Eigen::VectorXf D_full = Eigen::VectorXf::Zero(Nt*Nsd);

    //Time integration - Newmark Scheme
    float beta = 0.25;
    float gamma = 0.5;
    float dt = 0.01; //time step size
    int NT = 100; //number of time steps

    Eigen::LDLT<Eigen::MatrixXf> time_step_solver(MUU + (gamma*dt)*CUU + (beta*dt*dt)*KUU); //solver for time stepping

    std::vector<std::string> solution_files;
    std::vector<float> solution_timesteps;

    //Initial solution txt file
    std::ofstream D0_file("initial_solution.txt");
    for(int i = 0 ; i < Nt ; i++){
        D0_file << D0_full(3*i) << " " << D0_full(3*i + 1) << " " << D0_full(3*i + 2) << "\n";
    }

    for(int t = 0 ; t < NT ; t++){
        Eigen::VectorXf Ddot_predictor = Ddot0 + (1-gamma)*dt*Ddotdot0;
        Eigen::VectorXf D_predictor = D0 + dt*Ddot0 + ((0.5 - beta)*dt*dt)*Ddotdot0;
        Eigen::VectorXf rhs = F - CUU*Ddot_predictor - KUU*D_predictor;
        
        Eigen::VectorXf Ddotdot = time_step_solver.solve(rhs);
        Eigen::VectorXf Ddot = Ddot_predictor + (gamma*dt)*Ddotdot;
        Eigen::VectorXf DU = D_predictor + (beta*dt*dt)*Ddotdot;

        //construct full solution vector including known values at dirischlet boundary
        for(int i = 0 ; i < unknownIndexes.size() ; i++){
            D_full(unknownIndexes[i]) = DU(i);
        }
        for(int i = 0 ; i < dirischletIndexes.size() ; i++){
            D_full(dirischletIndexes[i]) = dirischletVal(i);
        }

        //Visualization - write solution at current time step to file
        std::string filename = "solutions/solution_" + std::to_string(t) + ".vtu";
        write_vtu(filename, nodes, elements, D_full);
        solution_files.push_back(filename);
        solution_timesteps.push_back(t*dt);

        //write txt files at some timesteps
        if(t == 2 || t == 5 || t == 7 || t == 10 || t == 12 || t == 15 || t == 17 || t == 20 || t == 25 || t == 50 || t ==75){
            std::string filename = "solutions/solution_" + std::to_string(t) + ".txt";
            std::ofstream D_file(filename);
            for(int i = 0 ; i < Nt ; i++){
                D_file << D_full(3*i) << " " << D_full(3*i + 1) << " " << D_full(3*i + 2) << "\n";
            }
        }

        //update for next time step
        D0 = DU;
        Ddot0 = Ddot;
        Ddotdot0 = Ddotdot;
    }

    // Write simulation pvd file
    write_pvd("final_solution.pvd", solution_files, solution_timesteps);

    //write solution to file
    std::ofstream D_file("final_solution.txt");
    for(int i = 0 ; i < Nt ; i++){
        D_file << D_full(3*i) << " " << D_full(3*i + 1) << " " << D_full(3*i + 2) << "\n";
    }

}