/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "3rdparty/INIReader.h"
#include "src/fastertransformer/models/llama/Llama.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"
#include "src/fastertransformer/utils/word_list.h"

#include <cuda_profiler_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

using namespace fastertransformer;

template<typename T>
void llama_example(const INIReader reader);

int read_start_ids(size_t            num_samples,
                   size_t            batch_size,
                   std::vector<int>* v_start_lengths,
                   std::vector<int>* v_start_ids,
                   std::vector<int>* output_lengths,
                   size_t&           max_input_len,
                   const int         end_id,
                   const int         beam_width,
                   std::string       file_name);

int main(int argc, char* argv[])
{
    mpi::initialize(&argc, &argv);
    srand(0);

    std::string ini_name;
    if (argc == 2) {
        ini_name = std::string(argv[1]);
    }
    else {
        ini_name = "../examples/cpp/llama/llama_config.ini";
    }

    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        return -1;
    }
    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");

    if (data_type == "fp32") {
        llama_example<float>(reader);
    }
    else if (data_type == "fp16") {
        llama_example<half>(reader);
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        llama_example<__nv_bfloat16>(reader);
    }
#endif
    else {
        FT_LOG_ERROR("is_fp16 should be 0 (use float) or 1 (use half).");
        return -1;
    }
    mpi::finalize();
    return 0;
}

template<typename T>
void llama_example(const INIReader reader)
{
    const std::string model_name = reader.Get("ft_instance_hyperparameter", "model_name");
    std::string       model_dir  = std::string(reader.Get("ft_instance_hyperparameter", "model_dir"));

    int tensor_para_size   = reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
    int pipeline_para_size = reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size");
    int int8_mode  = reader.GetInteger("ft_instance_hyperparameter", "int8_mode", 0);

    const size_t head_num             = reader.GetInteger(model_name, "head_num");
    const size_t size_per_head        = reader.GetInteger(model_name, "size_per_head");
    const size_t vocab_size           = reader.GetInteger(model_name, "vocab_size");
    const size_t decoder_layers       = reader.GetInteger(model_name, "num_layer");
    const size_t rotary_embedding_dim = reader.GetInteger(model_name, "rotary_embedding");
    const float  layernorm_eps        = reader.GetFloat(model_name, "layernorm_eps");
    const int    start_id             = reader.GetInteger(model_name, "start_id");
    const int    end_id               = reader.GetInteger(model_name, "end_id");

    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size   = reader.GetInteger(model_name, "inter_size");

    const size_t beam_width                 = reader.GetInteger("request", "beam_width");
    const uint   top_k                      = (uint)reader.GetInteger("request", "top_k");
    const float  top_p                      = reader.GetFloat("request", "top_p");
    const float  temperature                = reader.GetFloat("request", "temperature");
    const float  repetition_penalty         = reader.GetFloat("request", "repetition_penalty", 1.0f);
    const float  presence_penalty           = reader.GetFloat("request", "presence_penalty", 0.0f);
    const float  len_penalty                = reader.GetFloat("request", "len_penalty");
    const float  beam_search_diversity_rate = reader.GetFloat("request", "beam_search_diversity_rate");
    const int    min_length                 = reader.GetInteger("request", "min_length", 0);
    const size_t request_batch_size         = reader.GetInteger("request", "request_batch_size");
    const size_t num_samples                = reader.GetInteger("request", "num_samples");
    // The length of tokens we hope this model to generate
    // const int request_output_len = reader.GetInteger("request", "request_output_len");

    FT_CHECK(head_num % tensor_para_size == 0);
    FT_CHECK(decoder_layers % pipeline_para_size == 0);
    FT_CHECK_WITH_INFO(
        repetition_penalty == 1.0f || presence_penalty == 0.0f,
        fmtstr("Found ambiguous parameters repetition_penalty (%f) and presence_penalty (%f) "
               "which are mutually exclusive. Please remove one of repetition_penalty or presence_penalty "
               "or set to a default value.",
               repetition_penalty,
               presence_penalty));

    // Prepare the parallelism parameters
    int rank       = mpi::getCommWorldRank();
    int world_size = mpi::getCommWorldSize();
    if (rank == 0) {
        printf("Total ranks: %d.\n", world_size);
    }
    int device, device_count;
    check_cuda_error(cudaGetDeviceCount(&device_count));
    check_cuda_error(cudaSetDevice(rank % device_count));
    check_cuda_error(cudaGetDevice(&device));

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    printf("Device %s\n", prop.name);

    printf("P%d is running with GPU #%d.\n", rank, device);
    if (tensor_para_size * pipeline_para_size != world_size) {
        if (world_size % pipeline_para_size) {
            printf("[ERROR] tensor_para_size * pipeline_para_size should equal to world_size \n");
            exit(-1);
        }
        tensor_para_size = world_size / pipeline_para_size;
        printf("[INFO] Setting tensor_para_size to %d \n", tensor_para_size);
    }

    const int layers_per_group = decoder_layers / pipeline_para_size;
    if (layers_per_group * pipeline_para_size != (int)decoder_layers) {
        printf("[ERROR] layers_per_group (%d) * pipeline_para_size (%d) should equal to decoder_layers (%ld) \n",
               layers_per_group,
               pipeline_para_size,
               decoder_layers);
        exit(-1);
    }

    // assume gpu_num = k * n,
    // tensor parallelism group size is n
    // pipeline parallelism group size is k
    NcclParam tensor_para;
    NcclParam pipeline_para;
    ftNcclInitialize(tensor_para, pipeline_para, tensor_para_size, pipeline_para_size);

    // Handle bad_words dictionary
    std::vector<int> bad_words;
    read_word_list("../examples/cpp/llama/bad_words.csv", bad_words);

    int* d_bad_words = nullptr;
    deviceMalloc(&d_bad_words, bad_words.size(), false);
    cudaH2Dcpy(d_bad_words, bad_words.data(), bad_words.size());

    // Handle stop_words dictionary
    std::vector<int> stop_words;
    read_word_list("../examples/cpp/llama/stop_words.csv", stop_words);

    const size_t stop_words_len = stop_words.size() / 2;
    // Tile with same dict for each element
    std::vector<int> tiled_stop_words;
    for (int i = 0; i < request_batch_size; i++) {
        tiled_stop_words.insert(tiled_stop_words.end(), stop_words.begin(), stop_words.end());
    }

    
    int* d_stop_words = nullptr;
    deviceMalloc(&d_stop_words, tiled_stop_words.size(), false);
    cudaH2Dcpy(d_stop_words, tiled_stop_words.data(), tiled_stop_words.size());

    // Read ids of request from file.
    size_t           max_input_len = -1;
    std::vector<int> v_start_lengths;
    std::vector<int> v_start_ids;
    std::vector<int> output_lengths;
    read_start_ids(num_samples,
                   request_batch_size,
                   &v_start_lengths,
                   &v_start_ids,
                   &output_lengths,
                   max_input_len,
                   end_id,
                   1,
                   "../examples/cpp/llama/start_ids_request.txt");


    std::vector<int> start_ids(request_batch_size, start_id);
    std::vector<int> end_ids(request_batch_size, end_id);

    // Prompt Learning Configurations
    // NOTE: if you don't need prefix prompts, remember to set max_prefix_len to 0 and others to nullptr
    int prompt_learning_start_id = reader.GetInteger(model_name, "prompt_learning_start_id", end_id + 1);
    fastertransformer::PromptLearningType prompt_learning_type =
        static_cast<fastertransformer::PromptLearningType>(reader.GetInteger(model_name, "prompt_learning_type", 0));

    // NOTE: specify task names, take name id, prompt length in order to load those prompt learning tables.
    // NOTE: Please make sure task ids are continuous and start from 0
    // for example:
    // std::map<std::string, std::pair<int, int>> prefix_prompt_table_pair{{"no_prompt", {0, 0}},
    //                                                                     {"prompt_1", {1, 1}},
    //                                                                     {"prompt_2", {2, 2}},
    //                                                                     {"prompt_3", {3, 3}},
    //                                                                     {"prompt_4", {4, 4}},
    //                                                                     {"prompt_5", {5, 5}}};

    std::map<std::string, std::pair<int, int>> prefix_prompt_table_pair;

    // NOTE: get prompt table pairs from configuration files
    const int num_tasks = reader.GetInteger(model_name, "num_tasks", 0);
    for (int task_name_id = 0; task_name_id < num_tasks; task_name_id++) {
        std::string config_task_name = model_name + "_task_" + std::to_string(task_name_id);
        std::string task_name        = reader.Get(config_task_name, "task_name");
        const int   prompt_length    = reader.GetInteger(config_task_name, "prompt_length", 0);
        prefix_prompt_table_pair.insert({task_name, {task_name_id, prompt_length}});
    }

    // NOTE: task_name_ids for each sequence in one batch
    // Each sequence can have different prompt learning task ids
    std::vector<int> prefix_prompt_task_ids(request_batch_size, 0);

    // Set different task ids
    for (int i = 0; i < request_batch_size; i++) {
        prefix_prompt_task_ids[i] = (num_tasks > 0) ? i % num_tasks : 0;
    }

    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in");

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex*     cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
    if (std::is_same<T, half>::value) {
        cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    const bool                          use_gptj_residual = false;
    fastertransformer::LlamaWeight<T> gpt_weights(hidden_units,
                                                  inter_size,
                                                  vocab_size,
                                                  decoder_layers,
                                                  0,  // max_seq_len, deprecated
                                                  tensor_para.world_size_,
                                                  tensor_para.rank_,
                                                  pipeline_para.world_size_,
                                                  pipeline_para.rank_,
                                                  use_gptj_residual,
                                                  int8_mode,
                                                  prompt_learning_type,
                                                  prefix_prompt_table_pair);

    gpt_weights.loadModel(model_dir);
    unsigned long long random_seed;
    if (rank == 0) {
        random_seed = (unsigned long long)(0);
    }
    if (world_size > 1) {
        mpi::bcast(&random_seed, 1, mpi::MPI_TYPE_UNSIGNED_LONG_LONG, 0, mpi::COMM_WORLD);
    }

    AttentionType attention_type = getAttentionType<T>(size_per_head,
                                                       getSMVersion(),
                                                       true,   // remove_padding
                                                       0,      // gpt supports any-seq-length fmha
                                                       true,   // is_fuse
                                                       false,  // with_relative_position_bias
                                                       true);  // causal_mask

    Llama<T> gpt = Llama<T>(head_num,
                            size_per_head,
                            inter_size,
                            decoder_layers,
                            vocab_size,
                            rotary_embedding_dim,
                            layernorm_eps,
                            start_id,
                            end_id,
                            prompt_learning_start_id,
                            prompt_learning_type,
                            use_gptj_residual,
                            0.0f,
                            top_k,
                            top_p,
                            random_seed,
                            temperature,
                            len_penalty,
                            repetition_penalty,
                            tensor_para,
                            pipeline_para,
                            stream,
                            &cublas_wrapper,
                            &allocator,
                            false,
                            &prop,
                            attention_type,
                            int8_mode,
                            nullptr,
                            0,
                            1.0f);

    cudaProfilerStart();

    // test time
    struct timeval start, end;
    mpi::barrier();
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);

    ft_nvtx::setScope("total_time");
    PUSH_RANGE("total time")

    int prompt_num_tokens = 0;
    int total_num_tokens = 0;
    for (int i = 0; i < num_samples; i ++) {
        prompt_num_tokens += v_start_lengths[i];
        total_num_tokens += v_start_lengths[i] + output_lengths[i];
    }

    for (int i = 0; i < num_samples; i ++) {
        int* d_input_ids;
        int* d_input_lengths;
        deviceMalloc(&d_input_ids, request_batch_size * v_start_lengths[i], false);
        deviceMalloc(&d_input_lengths, request_batch_size, false);
        cudaH2Dcpy(d_input_ids, &v_start_ids[i], request_batch_size * v_start_lengths[i]);
        cudaH2Dcpy(d_input_lengths, &v_start_lengths[i], request_batch_size);

        int* d_output_ids;
        int* d_sequence_lengths;

        const int total_output_len = v_start_lengths[i] + output_lengths[i];
        total_num_tokens += total_output_len;

        printf("total output len: %d\n", total_output_len);
        printf("output len: %d\n", output_lengths[i]);

        deviceMalloc(&d_output_ids, request_batch_size * beam_width * total_output_len, false);
        deviceMalloc(&d_sequence_lengths, request_batch_size * beam_width, false);

        std::vector<uint32_t>                   output_seq_len(request_batch_size, total_output_len);
        std::unordered_map<std::string, Tensor> input_tensors = std::unordered_map<std::string, Tensor>{
            {"input_ids",
            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, (size_t)v_start_lengths[i]}, d_input_ids}},
            {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, d_input_lengths}},
            // NOTE: if you need prefix prompts, remember to add prefix_prompt_task_ids here
            // {"prompt_learning_task_name_ids", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{request_batch_size},
            // prefix_prompt_task_ids.data()}},
            {"output_seq_len",
            Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len.data()}},
            {"bad_words_list", Tensor{MEMORY_GPU, TYPE_INT32, {2, bad_words.size() / 2}, d_bad_words}},
            {"stop_words_list", Tensor{MEMORY_GPU, TYPE_INT32, {request_batch_size, 2, stop_words_len}, d_stop_words}},
            {"temperature", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &temperature}},
            {"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &len_penalty}},
            {"min_length", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &min_length}},
            {"start_id", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, start_ids.data()}},
            {"end_id", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, end_ids.data()}}};

        if (repetition_penalty != 1.0f) {
            input_tensors.insert(
                {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
        }
        if (presence_penalty != 0.0f) {
            input_tensors.insert(
                {"presence_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &presence_penalty}});
        }

        if (num_tasks > 0) {
            // Prefix Prompt Task Name Ids here
            input_tensors.insert(
                {"prompt_learning_task_name_ids",
                Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{request_batch_size}, prefix_prompt_task_ids.data()}});
        }

        if (top_k == 0 && top_p == 0.0f) {
            FT_CHECK(beam_width > 1);
            input_tensors.insert({"beam_search_diversity_rate",
                                Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
        }
        else {
            input_tensors.insert({"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});
            if (top_p != 0.0f) {
                input_tensors.insert({"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &top_p}});
            }
            if (top_k != 0) {
                input_tensors.insert({"runtime_top_k", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, &top_k}});
            }
        }

        std::unordered_map<std::string, Tensor> output_tensors = std::unordered_map<std::string, Tensor>{
            {"output_ids",
            Tensor{MEMORY_GPU,
                    TYPE_INT32,
                    std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                    d_output_ids}},
            {"sequence_length",
            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, beam_width}, d_sequence_lengths}},
            {"output_log_probs",
            Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{(size_t)output_lengths[i], request_batch_size, beam_width},
                    nullptr}}};

        print_mem_usage();

        cudaDeviceSynchronize();
        mpi::barrier();

        gpt.forward(&output_tensors, &input_tensors, &gpt_weights);

        cudaDeviceSynchronize();
        mpi::barrier();

        if (d_input_ids != nullptr) {
            cudaFree(d_input_ids);
        }
        if (d_input_lengths != nullptr) {
            cudaFree(d_input_lengths);
        }
        if (d_output_ids != nullptr) {
            deviceFree(d_output_ids);
        }
        if (d_sequence_lengths != nullptr) {
            deviceFree(d_sequence_lengths);
        }
    }

    POP_RANGE;
    ft_nvtx::resetScope();
    gettimeofday(&end, NULL);

    cudaProfilerStop();

    // if (rank == 0) {

    //     std::string fName   = "out";
    //     auto        outFile = std::ofstream(fName, std::ios::out);
    //     if (!outFile.is_open()) {
    //         printf("[WARNING] Cannot write results into output file %s \n", fName.c_str());
    //     }
    //     else {
    //         size_t outCount = total_output_len * request_batch_size * beam_width;
    //         int*   hBuf     = new int[outCount];

    //         cudaD2Hcpy(hBuf, d_output_ids, outCount);

    //         {
    //             std::cout << "Writing " << outCount << " elements\n";
    //             int zeroCount = 0;
    //             for (size_t i = 0; i < outCount; i++) {
    //                 if (hBuf[i] == int(0)) {
    //                     zeroCount++;
    //                 }
    //                 outFile << hBuf[i] << ", ";
    //                 if ((i + 1) % (total_output_len) == 0) {
    //                     outFile << std::endl;
    //                 }

    //                 if (i < 10) {
    //                     printf("%5d ", hBuf[i]);
    //                 }
    //                 if ((i + 1) % (total_output_len) == 0 && i < 10) {
    //                     std::cout << std::endl;
    //                 }
    //             }
    //             std::cout << std::endl << "zeroCount = " << zeroCount << std::endl;
    //         }
    //         delete[] hBuf;
    //     }
    // }

    printf("[INFO] request_batch_size %ld beam_width %ld head_num %ld size_per_head %ld total_output_len %d"
           " decoder_layers %ld vocab_size %ld FT-CPP-decoding-beamsearch-time %.2f ms\n",
           request_batch_size,
           beam_width,
           head_num,
           size_per_head,
           total_num_tokens,
           decoder_layers,
           vocab_size,
           ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / num_samples);
    
    double total_time = ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / 1000;
    printf("Throughput: %.2lf request/s\n", (double)(num_samples) / total_time);
    printf("Tokens/s: %.2lf tokens/s\n", (double)(total_num_tokens) / total_time);
    printf("Prompt_num_tokens: %d\n", prompt_num_tokens);
    printf("Total_num_tokens: %d\n", total_num_tokens);

    ftNcclParamDestroy(tensor_para);
    ftNcclParamDestroy(pipeline_para);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    cudaFree(d_bad_words);
    cudaFree(d_stop_words);
    return;
}

int read_start_ids(size_t            num_samples,
                   size_t            batch_size,
                   std::vector<int>* v_start_lengths,
                   std::vector<int>* v_start_ids,
                   std::vector<int>* output_lengths,
                   size_t&           max_input_len,
                   const int         end_id,
                   const int         beam_width,
                   std::string       file_name)
{
    std::vector<std::vector<int>> tmp_start_ids;
    std::vector<int>              tmp_start_lengths;
    std::vector<int>              tmp_output_lengths;

    std::ifstream start_id_file(file_name, std::ios::in);
    int           line_num = 0;
    if (start_id_file.is_open()) {
        std::string line;
        while (std::getline(start_id_file, line)) {
            std::stringstream lineStream(line);
            std::string       vals;
            int               i1 = 0;
            std::vector<int>  tmp_vec;
            while (std::getline(lineStream, vals, ',')) {
                tmp_vec.push_back(std::stoi(vals));
                i1++;
            }

            std::getline(start_id_file, line);
            int prompt_length = std::stoi(line);
            // printf("%d ", prompt_length);
            std::getline(start_id_file, line);
            int output_length = std::stoi(line);
            // printf("%d ", output_length);

            // for (int e : tmp_vec) {
            //     printf("%d, ", e);
            // }
            // printf("\n");

            tmp_start_ids.push_back(tmp_vec);
            tmp_start_lengths.push_back(i1);
            tmp_output_lengths.push_back(output_length);
            line_num++;
        }
        if (batch_size == 0) {
            batch_size = line_num;
        }
    }
    else {
        printf("[WARNING] Cannot open the file '%s'. \n", file_name.c_str());
        max_input_len = 0;
        return 0;
    }

    max_input_len = tmp_start_lengths.data()[0];
    for (uint i = 1; i < (uint)tmp_start_lengths.size(); i++) {
        max_input_len = max_input_len > tmp_start_lengths.data()[i] ? max_input_len : tmp_start_lengths.data()[i];
    }

    while ((int)tmp_start_lengths.size() < batch_size) {
        std::vector<int> padding_ids;
        for (int i = 0; i < max_input_len; i++) {
            padding_ids.push_back(end_id);
        }
        tmp_start_ids.push_back(padding_ids);
        tmp_start_lengths.push_back(max_input_len);
    }

    // Add padding
    for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
        for (int j = (int)tmp_start_ids[i].size(); j < max_input_len; j++) {
            tmp_start_ids[i].push_back(end_id);
        }
    }

    for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
        for (int b = 0; b < beam_width; b++) {
            for (int j = 0; j < (int)tmp_start_ids[i].size(); j++) {
                v_start_ids->push_back(tmp_start_ids[i][j]);
            }
            v_start_lengths->push_back(tmp_start_lengths[i]);
            output_lengths->push_back(tmp_output_lengths[i]);
        }
    }
    return batch_size;
}