#include "kernel_common.h"
#include <cstdlib>
#include <iostream>

extern "C" int CliPPParseFlag(const char*,int*);
extern "C" int CliPPMultiplicityVersion(){return 3;}
extern "C" int CliPPChainStatus(int count,int* alt,int* depth,int* major,int* total,
    double purity,double* pilot_cp,int* requested_k,int k_count,char* output)
{
    int required=-1,forced=-1;
    if(CliPPParseFlag(std::getenv("CLIPP_REQUIRE_CUDA"),&required) ||
       CliPPParseFlag(std::getenv("CLIPP_FORCE_CPU"),&forced)){
        std::cerr<<"Invalid CLIPP Boolean environment flag."<<std::endl;return kCliPPError;
    }
    const bool require_cuda=required==1;
    if(require_cuda && forced==1){
        std::cerr<<"Conflicting CLIPP_REQUIRE_CUDA and CLIPP_FORCE_CPU."<<std::endl;
        return kCliPPError;
    }
#ifdef USE_CUDA
    if(forced!=1){
        const int status=CliPPChainCUDA(count,alt,depth,major,total,purity,pilot_cp,requested_k,k_count,output);
        if(status!=kCudaUnavailable) return status;
        if(require_cuda) return status;
        std::cerr<<"CUDA unavailable before chain work; selecting the CPU backend."<<std::endl;
    }
#endif
    if(require_cuda){
        std::cerr<<"CUDA is required; refusing CPU fallback."<<std::endl;
        return kCudaUnavailable;
    }
    return CliPPChainCPU(count,alt,depth,major,total,purity,pilot_cp,requested_k,k_count,output);
}
// Old complete-graph/SCAD callers must migrate; they cannot execute silently.
extern "C" int CliPPStatus(...){
    std::cerr<<"Legacy complete-graph CliPPStatus is retired; use CliPPChainStatus and rebuild the wrapper."<<std::endl;
    return kCliPPError;
}
extern "C" void CliPP(...){(void)CliPPStatus();}
#ifndef USE_CUDA
extern "C" int CliPPWarmupCUDA(){return kCudaUnavailable;}
#endif
