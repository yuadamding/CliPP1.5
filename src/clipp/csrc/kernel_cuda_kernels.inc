R"CUDA(
#define CLIPP_INF (__longlong_as_double(0x7ff0000000000000LL))
struct MultiplicityLikelihood { double nll; double gradient; double curvature; };
__device__ __forceinline__ MultiplicityLikelihood multiplicity_likelihood_x(
    double x, double r, double n, int major, double scale, double log_choose)
{
    double maximum = -CLIPP_INF;
    for(int m = 1; m <= major; ++m){
        const double a = fmin(1.0, scale * m);
        const double p = fmin(1.0, x * a);
        const double log_p = x > 0.0 ? fmin(0.0,log(x) + log(scale) + log(double(m))) : -CLIPP_INF;
        const double ell = (r > 0.0 ? r * log_p : 0.0) + (n > r ? (n-r) * log1p(-p) : 0.0);
        maximum = fmax(maximum, ell);
    }
    if(!isfinite(maximum)) return {CLIPP_INF, 0.0, 1.0};
    double mass = 0.0, gradient = 0.0, curvature = 0.0, boundary_gradient = 0.0;
    bool infinite_boundary_curvature=false;
    for(int m = 1; m <= major; ++m){
        const double a = fmin(1.0, scale * m);
        const double p = fmin(1.0, x * a);
        const double log_p = x > 0.0 ? fmin(0.0,log(x) + log(scale) + log(double(m))) : -CLIPP_INF;
        const double ell = (r > 0.0 ? r * log_p : 0.0) + (n > r ? (n-r) * log1p(-p) : 0.0);
        if(!isfinite(ell)){
            // A vanishing binomial state can still have a nonzero one-sided
            // derivative. Optimizer iterates stay in the declared interior.
            if(p==1.0 && n-r==1.0){boundary_gradient+=a*exp(-maximum);infinite_boundary_curvature=true;}
            if(p==1.0 && n-r==2.0) curvature+=2.0*a*a*exp(-maximum);
            continue;
        }
        const double weight = exp(ell-maximum);
        const double g = (r > 0.0 ? -r/x : 0.0) + (n > r ? (n-r)*a/(1.0-p) : 0.0);
        const double h = (r > 0.0 ? r/(x*x) : 0.0) + (n > r ? (n-r)*a*a/((1.0-p)*(1.0-p)) : 0.0);
        mass += weight;
        gradient += weight*g;
        curvature += weight*h;
    }
    return {-maximum-log(mass)+log(double(major))-log_choose,
            (gradient+boundary_gradient)/mass,
            infinite_boundary_curvature?CLIPP_INF:fmax(1e-8, curvature/mass)};
}

extern "C" __global__ void chain_likelihood_kernel(int count,const double* x,
    const int* alt,const int* depth,const int* major,const double* scale,
    const double* log_choose,double* values,double* gradient,double* curvature)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const auto value=multiplicity_likelihood_x(x[i],alt[i],depth[i],major[i],scale[i],log_choose[i]);
    values[i]=value.nll;gradient[i]=value.gradient;curvature[i]=value.curvature;
}
)CUDA"
