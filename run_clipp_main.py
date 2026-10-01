import sys
import os

#if not (sys.version_info[0] == 3 and sys.version_info[1] >= 5 and sys.version_info[2] >= 1):
version_morph = sys.version_info[0]*10000+sys.version_info[1]*100+sys.version_info[2]
version_base = 30501
if not (version_morph >= version_base):
    sys.stderr.write("Error message: CliPP can only run with python >=3.5.1\n")
    sys.exit(-1)

import argparse
import ctypes
import glob
import subprocess
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "src"))

from run_kernel_nosub import run_clipp_nosub
from run_kernel_sub import run_clipp_sub
from penalty_selection import run_model_selection


CUDA_MIN_SNV_COUNT = 1000


def _find_clipp_library():
    patterns = [
        os.path.join(current_dir, "CliPP*%s*.so" % (sys.platform)),
        os.path.join(current_dir, "build", "*", "CliPP*%s*.so" % (sys.platform)),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def warmup_cuda_if_available():
    if os.environ.get("CLIPP_FORCE_CPU"):
        return False

    clipp_lib_path = _find_clipp_library()
    if clipp_lib_path is None:
        print("CUDA warmup skipped; no CliPP shared library was found.")
        return False

    clipp_lib = ctypes.CDLL(clipp_lib_path)
    warmup = getattr(clipp_lib, "CliPPWarmupCUDA", None)
    if warmup is None:
        raise RuntimeError("Rebuild CliPP for the chain distance-to-set backend.")
    warmup.argtypes = []
    warmup.restype = ctypes.c_int
    print("Warming up CUDA...")
    status = warmup()
    if status == 0:
        print("CUDA warmup finished.")
        return True
    if status == 2:
        print("CUDA warmup skipped; no CUDA backend/device is available.")
        return False
    raise RuntimeError("CUDA chain warmup failed with status %s; refusing CPU fallback." % status)


def count_preprocessed_snvs(preprocess_dir):
    r_path = os.path.join(preprocess_dir, "r.txt")
    try:
        with open(r_path, "r") as handle:
            return sum(1 for line in handle if line.strip() != "")
    except OSError as err:
        raise RuntimeError("Cannot read preprocessed SNV count from %s: %s" % (r_path, err))


def select_backend_after_preprocess(preprocess_dir, device="auto"):
    snv_count = count_preprocessed_snvs(preprocess_dir)
    print("Preprocessed SNVs: %d" % snv_count)

    if device == "cuda" or (device == "auto" and os.environ.get("CLIPP_REQUIRE_CUDA")):
        if os.environ.get("CLIPP_FORCE_CPU"):
            raise RuntimeError("--device cuda conflicts with CLIPP_FORCE_CPU.")
        os.environ["CLIPP_REQUIRE_CUDA"] = "1"
        if not warmup_cuda_if_available():
            raise RuntimeError("--device cuda requires an available CUDA backend/device.")
        print("CliPP backend explicitly selected: CUDA.")
        return snv_count
    if device == "cpu":
        if os.environ.get("CLIPP_REQUIRE_CUDA"):
            raise RuntimeError("--device cpu conflicts with CLIPP_REQUIRE_CUDA.")
        os.environ["CLIPP_FORCE_CPU"] = "1"

    if os.environ.get("CLIPP_FORCE_CPU"):
        print("CliPP backend auto-selection: CPU (CLIPP_FORCE_CPU is set).")
        return snv_count

    if snv_count <= CUDA_MIN_SNV_COUNT:
        os.environ["CLIPP_FORCE_CPU"] = "1"
        print(
            "CliPP backend auto-selection: CPU "
            "(preprocessed SNVs <= %d; CUDA requires > %d)."
            % (CUDA_MIN_SNV_COUNT, CUDA_MIN_SNV_COUNT)
        )
        return snv_count

    print("CliPP backend auto-selection: checking CUDA (preprocessed SNVs > %d)." % CUDA_MIN_SNV_COUNT)
    if warmup_cuda_if_available():
        print("CliPP backend auto-selection: CUDA.")
    else:
        os.environ["CLIPP_FORCE_CPU"] = "1"
        print("CliPP backend auto-selection: CPU (CUDA is unavailable).")
    return snv_count


parser = argparse.ArgumentParser()

parser.add_argument("snv_input", type=str, help="Path of the snv input.")
parser.add_argument("cn_input", type=str, help="Path of the copy number input.")
parser.add_argument("purity_input", type=str, help="Path of the purity input.")
parser.add_argument("-i", "--sample_id", type=str, default="sample_id", help="Name of the sample being processed. Default is 'sample'.")
parser.add_argument("-b", "--subsampling", action='store_true', help="Whether doing subsampling or not. Default is not doing the subsampling, and a flag -b is needed when you want to do subsampling.")
cluster_options = parser.add_mutually_exclusive_group()
cluster_options.add_argument("--clusters", "--K", dest="clusters", type=int, help="Fit one fixed cluster capacity K (1 through 10). The distance-to-set constraint permits at most K contiguous clusters.")
cluster_options.add_argument("--max-clusters", type=int, default=10, help="Compare cluster capacities 1 through this value by BIC (default 10, maximum 10); mutually exclusive with --clusters.")
parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Execution backend; cuda requires CUDA even for small inputs and forbids CPU fallback.")
parser.add_argument("--center-refit", choices=("conditional",), default="conditional",
                    help="Final center estimator: conditional chain block fit (the only supported mode).")
parser.add_argument("-p", "--preprocess", type=str, default="preprocess_result/", help="Directory that stores the preprocess results. Default name is 'preprocess_result/'.")
parser.add_argument("-f", "--final", type=str, default="final_result/", help="Directory that stores the final results after postprocessing. Default name is 'final_result/'.")

parser.add_argument("-s", "--subsample_size", type=int, help="(Required if doing subsampling) The number of SNVs you want to include in each subsamples.")
parser.add_argument("-n", "--rep_num", type=int, help="(Required if doing subsampling) The number of random subsamples needed.")
parser.add_argument("-w", "--window_size", type=float, default=0.05, help="Controls the length of the window. Takes value between 0 and 1. Default is 0.05.")
parser.add_argument("-o", "--overlap_size", type=float, default=0.0, help="Controls the overlapped length of two consecutive windows. Takes value between 0 and 1. Default is 0.")

args = parser.parse_args()

run_preprocess = os.path.join(current_dir, "src/preprocess.R")
sample_id = args.sample_id
sample_id = sample_id.strip()
final_result = args.final.strip()

if sample_id == "":
	sys.stderr.write("User specified sample id is empty. Use default sample_id instead.\n")
	sample_id = "sample_id"

if final_result == "":
    sys.stderr.write("User specified final is empty. Use default final_result instead.\n")
    final_result = "final_result"

if args.subsampling:
    if args.subsample_size is None:
        sys.stderr.write("Please specify subsample_size\n")
        sys.exit(-1)

    if args.rep_num is None:
        sys.stderr.write("Please specify rep_num\n")
        sys.exit(-1)


if not 1 <= args.max_clusters <= 10:
    parser.error("--max-clusters must be between 1 and 10.")
if args.clusters is not None and not 1 <= args.clusters <= 10:
    parser.error("--clusters must be between 1 and 10.")

result_dir = os.path.join(current_dir, sample_id)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

path_for_preprocess = os.path.join(result_dir, args.preprocess)
path_for_preliminary = os.path.join(result_dir, "preliminary_result")
path_for_final = os.path.join(result_dir, final_result)

# Run preprocessing
print("Running preprocessing...")
p_preprocess = subprocess.Popen(["Rscript",
                                 run_preprocess,
                                 args.snv_input,
                                 args.cn_input,
                                 args.purity_input,
                                 args.sample_id,
                                 path_for_preprocess],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

_stdout, _stderr = p_preprocess.communicate()

if p_preprocess.returncode != 0:
    print(_stderr.decode().strip())
    sys.exit(-1)
print("Preprocessing finished.")
retained_count = select_backend_after_preprocess(path_for_preprocess, args.device)
if args.clusters is not None:
    if args.clusters > retained_count:
        parser.error("--clusters must not exceed the number of retained mutations (%d)." % retained_count)
    cluster_list = [args.clusters]
else:
    cluster_list = list(range(1, min(args.max_clusters, retained_count) + 1))
if args.subsampling and args.subsample_size < max(cluster_list):
    parser.error("Subsample size must support every requested cluster capacity; increase -s or reduce --clusters/--max-clusters.")

# The mixture model requires likelihood refits, replacing fixed-multiplicity
# averaging and purity-ratio heuristics in the legacy R postprocessor.
print("Running the main CliPP function...")
start = time.time()
if args.subsampling:
    run_clipp_sub(path_for_preprocess, path_for_preliminary, cluster_list,
                  args.subsample_size, args.rep_num, args.window_size, args.overlap_size)
else:
    run_clipp_nosub(path_for_preprocess, path_for_preliminary, cluster_list)
print("\nElapsed time: %.6fsec\n" % (time.time() - start))

# Refit the proposed chain blocks and select the minimum ordinary BIC.
run_model_selection(path_for_preprocess, path_for_preliminary, path_for_final,
                     cluster_list, reps=args.rep_num if args.subsampling else None,
                     center_refit=args.center_refit)
print("Main CliPP function finished.")
