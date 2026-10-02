// Small C ABI for identity checks and independent numerical regression tests.
#include "kernel_common.h"
#include <cctype>
#include <string>
#include <algorithm>
#include <iomanip>
#include <sstream>

#ifndef CLIPP_BUILD_ID
#error "Build through the package builder to bind a native identity"
#endif
extern "C" const char* CliPPBuildId(){return CLIPP_BUILD_ID;}
extern "C" const char* CliPPModelVersion(){return "uniform_1_to_major_v1";}
extern "C" const char* CliPPNumerics(){
    static const std::string config=[](){
        std::ostringstream out;
        out<<std::setprecision(17)<<"{\"levels\":"<<kChainLevels<<",\"iterations_per_level\":"<<kChainIterationsPerLevel
           <<",\"stationarity_tolerance\":"<<kChainStationarityTolerance<<",\"constraint_tolerance\":"<<kChainConstraintTolerance
           <<",\"maximum_rho\":"<<kChainMaximumRho<<",\"ccf_bounds\":["<<kChainLowerCCF<<','<<kChainUpperCCF<<"]}";
        return out.str();
    }();
    return config.c_str();
}
extern "C" int CliPPCUDACompiled(){
#ifdef USE_CUDA
    return 1;
#else
    return 0;
#endif
}

// -1 unspecified, 0 false, 1 true; status 1 rejects other spellings.
extern "C" int CliPPParseFlag(const char* value, int* result){
    if(!result) return 1;
    if(!value || !*value){*result=-1;return 0;}
    std::string text(value);
    std::transform(text.begin(),text.end(),text.begin(),[](unsigned char c){return std::tolower(c);});
    if(text=="1" || text=="true" || text=="yes" || text=="on"){*result=1;return 0;}
    if(text=="0" || text=="false" || text=="no" || text=="off"){*result=0;return 0;}
    return 1;
}
extern "C" int CliPPEvaluateCPU(int n,const int* r,const int* depth,const int* major,
    const int* total,double purity,const double* x,double* output){
    if(n<1 || !r || !depth || !major || !total || !x || !output || !std::isfinite(purity) || purity<=0 || purity>1) return 1;
    for(int i=0;i<n;++i){
        if(r[i]<0 || depth[i]<=0 || r[i]>depth[i] || major[i]<1 || total[i]<major[i] || !std::isfinite(x[i]) || x[i]<0 || x[i]>1) return 1;
        const double scale=purity/(2*(1-purity)+purity*total[i]);
        const double choose=std::lgamma(depth[i]+1.0)-std::lgamma(r[i]+1.0)-std::lgamma(depth[i]-r[i]+1.0);
        const auto v=multiplicity_likelihood_x(x[i],r[i],depth[i],major[i],scale,choose);
        output[3*i]=v.nll;output[3*i+1]=v.gradient;output[3*i+2]=v.curvature;
    }
    return 0;
}
extern "C" int CliPPProject(int n,const double* x,int k,double* z){
    if(n<1 || k<1 || k>std::min(n,10) || !x || !z) return 1;
    try {
        for(int i=0;i<n;++i) if(!std::isfinite(x[i])) return 1;
        const auto result=project_chain_jumps(std::vector<double>(x,x+n),k);
        std::copy(result.begin(),result.end(),z);return 0;
    } catch(...) {return 1;}
}
extern "C" int CliPPBoxQP(int n,const double* curvature,const double* rhs,const double* z,
    double rho,double lower,double upper,double* output){
    if(n<1 || !curvature || !rhs || !z || !output || !std::isfinite(rho) || rho<0 || !std::isfinite(lower) || !std::isfinite(upper) || lower>=upper) return 1;
    for(int i=0;i<n;++i) if(!std::isfinite(curvature[i]) || curvature[i]<=0 || !std::isfinite(rhs[i]) || (i<n-1 && !std::isfinite(z[i]))) return 1;
    try {
        bool solved=false;
        const auto result=boxed_chain_laplacian_quadratic(std::vector<double>(curvature,curvature+n),
            std::vector<double>(rhs,rhs+n),std::vector<double>(z,z+n-1),rho,&solved,lower,upper);
        std::copy(result.begin(),result.end(),output);return solved?0:4;
    } catch(...) {return 1;}
}
