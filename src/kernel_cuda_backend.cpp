#include "kernel_common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check_cuda(cudaError_t status,const char* action){
    if(status!=cudaSuccess)throw std::runtime_error(std::string(action)+": "+cudaGetErrorString(status));
}
void check_driver(CUresult status,const char* action){
    if(status!=CUDA_SUCCESS){const char* reason=nullptr;cuGetErrorString(status,&reason);throw std::runtime_error(std::string(action)+": "+(reason?reason:"CUDA driver error"));}
}
void check_nvrtc(nvrtcResult status,const char* action){
    if(status!=NVRTC_SUCCESS)throw std::runtime_error(std::string(action)+": "+nvrtcGetErrorString(status));
}
const char* cuda_source(){return
#include "kernel_cuda_kernels.inc"
;}
struct Program {
    CUmodule module=nullptr;
    CUfunction function=nullptr;
    Program(){
        cudaDeviceProp properties;
        check_cuda(cudaGetDeviceProperties(&properties,0),"Query CUDA device");
        check_cuda(cudaSetDevice(0),"Select CUDA device");
        check_cuda(cudaFree(nullptr),"Create CUDA primary context");
        // Native device code avoids requiring the driver's PTX JIT to support
        // the newer NVRTC toolchain (CUDA minor-version compatibility).
        const std::string architecture="--gpu-architecture=sm_"+std::to_string(properties.major)+std::to_string(properties.minor);
        nvrtcProgram program=nullptr;
        check_nvrtc(nvrtcCreateProgram(&program,cuda_source(),"clipp_chain.cu",0,nullptr,nullptr),"Create chain CUDA program");
        try{
            const char* options[]={"--std=c++17",architecture.c_str()};
            const auto status=nvrtcCompileProgram(program,2,options);
            if(status!=NVRTC_SUCCESS){
                size_t length=0;nvrtcGetProgramLogSize(program,&length);std::string log(length,'\0');nvrtcGetProgramLog(program,&log[0]);
                throw std::runtime_error("Compile chain CUDA program: "+log);
            }
            size_t length=0;check_nvrtc(nvrtcGetCUBINSize(program,&length),"Get chain CUBIN size");
            if(!length)throw std::runtime_error("NVRTC produced no native device code.");
            std::vector<char> cubin(length);check_nvrtc(nvrtcGetCUBIN(program,cubin.data()),"Get chain CUBIN");
            check_driver(cuModuleLoadDataEx(&module,cubin.data(),0,nullptr,nullptr),"Load chain CUDA module");
            check_driver(cuModuleGetFunction(&function,module,"chain_likelihood_kernel"),"Resolve chain likelihood kernel");
            nvrtcDestroyProgram(&program);
        }catch(...){nvrtcDestroyProgram(&program);if(module){cuModuleUnload(module);module=nullptr;}throw;}
    }
    ~Program(){if(module)cuModuleUnload(module);}
    Program(const Program&)=delete;
    Program& operator=(const Program&)=delete;
};
template<typename T> struct DeviceArray {
    T* pointer=nullptr;
    size_t count;
    explicit DeviceArray(size_t size):count(size){check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer),count*sizeof(T)),"Allocate linear chain buffer");}
    ~DeviceArray(){if(pointer)cudaFree(pointer);}
    void upload(const std::vector<T>& values){check_cuda(cudaMemcpy(pointer,values.data(),count*sizeof(T),cudaMemcpyHostToDevice),"Upload chain vector");}
    void download(std::vector<T>& values){values.resize(count);check_cuda(cudaMemcpy(values.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost),"Download chain likelihood");}
    DeviceArray(const DeviceArray&)=delete;
    DeviceArray& operator=(const DeviceArray&)=delete;
};
bool device_available(){
    int count=0;const auto status=cudaGetDeviceCount(&count);
    return status==cudaSuccess && count>0;
}
}

extern "C" int CliPPWarmupCUDA(){
    if(!device_available())return kCudaUnavailable;
    try{Program program;return kCliPPOk;}
    catch(const std::exception& error){std::cerr<<"CUDA chain warmup failed: "<<error.what()<<std::endl;return kCudaFailedAfterWork;}
}

int CliPPChainCUDA(int count,int* alt,int* depth,int* major,int* total,
    double purity,double* pilot_cp,int* requested_k,int k_count,char* output)
{
    try{
        const auto data=prepare_chain_inputs(count,alt,depth,major,total,purity,pilot_cp,requested_k,k_count,output);
        if(!device_available())return kCudaUnavailable;
        Program program;
        DeviceArray<int> d_alt(count),d_depth(count),d_major(count);
        DeviceArray<double> d_scale(count),d_log_choose(count),d_x(count),d_values(count),d_gradient(count),d_curvature(count);
        d_alt.upload(data.alt);d_depth.upload(data.depth);d_major.upload(data.major);
        d_scale.upload(data.scale);d_log_choose.upload(data.log_choose);
        ChainEvaluator evaluate=[&](const std::vector<double>& x,std::vector<double>& values,
            std::vector<double>& gradient,std::vector<double>& curvature){
            d_x.upload(x);
            void* arguments[]={&count,&d_x.pointer,&d_alt.pointer,&d_depth.pointer,&d_major.pointer,
                &d_scale.pointer,&d_log_choose.pointer,&d_values.pointer,&d_gradient.pointer,&d_curvature.pointer};
            check_driver(cuLaunchKernel(program.function,(count+255)/256,1,1,256,1,1,0,nullptr,arguments,nullptr),"Evaluate chain multiplicity likelihood");
            d_values.download(values);d_gradient.download(gradient);d_curvature.download(curvature);
        };
        return run_chain_candidates(data,requested_k,k_count,output,evaluate,"cuda_likelihood_host_chain");
    }catch(const std::invalid_argument& error){
        std::cerr<<"CliPP chain input error: "<<error.what()<<std::endl;return kCliPPError;
    }catch(const std::exception& error){
        std::cerr<<"CliPP chain CUDA failure; no CPU fallback: "<<error.what()<<std::endl;return kCudaFailedAfterWork;
    }
}
