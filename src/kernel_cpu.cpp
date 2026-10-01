#include "kernel_common.h"
#include <iostream>

int CliPPChainCPU(int count,int* alt,int* depth,int* major,int* total,
    double purity,double* pilot_cp,int* requested_k,int k_count,char* output)
{
    try{
        const auto data=prepare_chain_inputs(count,alt,depth,major,total,purity,pilot_cp,requested_k,k_count,output);
        ChainEvaluator evaluate=[&](const std::vector<double>& x,std::vector<double>& values,
            std::vector<double>& gradient,std::vector<double>& curvature){
            values.resize(count);gradient.resize(count);curvature.resize(count);
            for(int i=0;i<count;++i){
                const auto value=multiplicity_likelihood_x(x[i],data.alt[i],data.depth[i],data.major[i],data.scale[i],data.log_choose[i]);
                values[i]=value.nll;gradient[i]=value.gradient;curvature[i]=value.curvature;
            }
        };
        return run_chain_candidates(data,requested_k,k_count,output,evaluate,"cpu");
    }catch(const std::exception& error){
        std::cerr<<"CliPP chain CPU failure: "<<error.what()<<std::endl;
        return kCliPPError;
    }
}
