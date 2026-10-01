#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <vector>

constexpr int kCliPPOk = 0;
constexpr int kCliPPError = 1;
constexpr int kCudaUnavailable = 2;
constexpr int kCudaFailedAfterWork = 3;
constexpr double kChainLowerCCF = 1e-8;
constexpr double kChainUpperCCF = 1.0-1e-8;

struct MultiplicityLikelihood {
    double nll;
    double gradient;
    double curvature;
};

// x is CCF, so p_m = x * purity * m / (2*(1-purity)+purity*total_cn).
// Curvature is the positive posterior mean complete-data observed curvature;
// the actual marginalized objective is checked by line search.
inline MultiplicityLikelihood multiplicity_likelihood_x(
    double x, double r, double n, int major, double scale, double log_choose)
{
    double maximum = -INFINITY;
    for(int m = 1; m <= major; ++m){
        const double a = std::fmin(1.0, scale * m);
        const double p = std::fmin(1.0, x * a);
        const double log_p = x > 0.0 ? std::fmin(0.0,std::log(x) + std::log(scale) + std::log(double(m))) : -INFINITY;
        const double ell = (r > 0.0 ? r * log_p : 0.0) + (n > r ? (n-r) * std::log1p(-p) : 0.0);
        maximum = std::fmax(maximum, ell);
    }
    if(!std::isfinite(maximum)) return {INFINITY, 0.0, 1.0};
    double mass = 0.0, gradient = 0.0, curvature = 0.0, boundary_gradient = 0.0;
    bool infinite_boundary_curvature=false;
    for(int m = 1; m <= major; ++m){
        const double a = std::fmin(1.0, scale * m);
        const double p = std::fmin(1.0, x * a);
        const double log_p = x > 0.0 ? std::fmin(0.0,std::log(x) + std::log(scale) + std::log(double(m))) : -INFINITY;
        const double ell = (r > 0.0 ? r * log_p : 0.0) + (n > r ? (n-r) * std::log1p(-p) : 0.0);
        if(!std::isfinite(ell)){
            // A vanishing binomial state can still have a nonzero one-sided
            // derivative. Optimizer iterates stay in the declared interior.
            if(p==1.0 && n-r==1.0){boundary_gradient+=a*std::exp(-maximum);infinite_boundary_curvature=true;}
            if(p==1.0 && n-r==2.0) curvature+=2.0*a*a*std::exp(-maximum);
            continue;
        }
        const double weight = std::exp(ell-maximum);
        const double g = (r > 0.0 ? -r/x : 0.0) + (n > r ? (n-r)*a/(1.0-p) : 0.0);
        const double h = (r > 0.0 ? r/(x*x) : 0.0) + (n > r ? (n-r)*a*a/((1.0-p)*(1.0-p)) : 0.0);
        mass += weight;
        gradient += weight*g;
        curvature += weight*h;
    }
    return {-maximum-std::log(mass)+std::log(double(major))-log_choose,
            (gradient+boundary_gradient)/mass,
            infinite_boundary_curvature?INFINITY:std::fmax(1e-8, curvature/mass)};
}

struct ChainData {
    int count;
    double purity;
    std::vector<int> alt, depth, major;
    std::vector<double> scale, log_choose, pilot_x;
};

using ChainEvaluator = std::function<void(const std::vector<double>&,
    std::vector<double>&, std::vector<double>&, std::vector<double>&)>;

ChainData prepare_chain_inputs(int count, const int* alt, const int* depth,
    const int* major, const int* total, double purity, const double* pilot_cp,
    const int* requested_k, int k_count, const char* output);

// Deterministic Euclidean projection onto {z: ||z||_0 <= K-1}; ties use
// frozen chain edge index. Work and storage are O(N), since K <= 10.
std::vector<double> project_chain_jumps(const std::vector<double>& x, int k);
std::vector<int> chain_labels(const std::vector<double>& projected);
std::vector<double> boxed_tridiagonal_quadratic(const std::vector<double>& diagonal,
    const std::vector<double>& rhs, double off_diagonal, bool* solved = nullptr,
    double lower=0.0, double upper=1.0);
// Fix only the support of the sparse projection. Its nonzero edges are free:
// the projected values can follow D*x, so no quadratic coupling crosses them.
double chain_support_penalty(const std::vector<double>& x,
    const std::vector<double>& projected, double rho);
std::vector<double> boxed_chain_support_quadratic(const std::vector<double>& diagonal,
    const std::vector<double>& rhs, const std::vector<double>& projected,
    double rho, bool* solved = nullptr, double lower=0.0, double upper=1.0);
// As above, but receive the likelihood curvature separately from rho*D'D.
// Positive effective-curvature elimination avoids cancellation at large rho.
std::vector<double> boxed_chain_laplacian_quadratic(const std::vector<double>& curvature,
    const std::vector<double>& rhs, const std::vector<double>& projected,
    double rho, bool* solved = nullptr, double lower=0.0, double upper=1.0);

int run_chain_candidates(const ChainData& data, const int* requested_k, int k_count,
    const char* output, const ChainEvaluator& evaluate, const char* backend);
int CliPPChainCPU(int count, int* alt, int* depth, int* major, int* total,
    double purity, double* pilot_cp, int* requested_k, int k_count, char* output);
#ifdef USE_CUDA
int CliPPChainCUDA(int count, int* alt, int* depth, int* major, int* total,
    double purity, double* pilot_cp, int* requested_k, int k_count, char* output);
#endif
