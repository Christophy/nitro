#include "sdCPP.h"

#include <context/llama_server_context.h>

#include "common.h"
#include "stable-diffusion.cpp/stable-diffusion.h"
#include "utils/nitro_utils.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "common/stb_image_write.h"

using namespace inferences;
using json = nlohmann::json;


struct sd_params {
  int n_threads = -1;

  std::string model_path;
  std::string vae_path;
  std::string taesd_path = "/app/nitro/build/taesd.safetensors";
  std::string esrgan_path;
  ggml_type wtype = GGML_TYPE_COUNT;
  std::string lora_model_dir;
  std::string output_path = "output.png";
  std::string input_path;

  std::string prompt;
  std::string negative_prompt;
  float cfg_scale = 1.0f;
  int clip_skip   = -1;  // <= 0 represents unspecified
  int width       = 512;
  int height      = 512;
  int batch_count = 1;

  sample_method_t sample_method = EULER_A;
  schedule_t schedule          = DEFAULT;
  // make defualt steps to 4, to apply at least for sd-lightning
  int sample_steps           = 4;
  float strength             = 0.75f;
  rng_type_t rng_type        = CUDA_RNG;
  int64_t seed               = 42;
  bool verbose               = false;
  bool vae_tiling            = false;
};


struct sd_ctx_t* sd;




static std::string sd_basename(const std::string& path) {
  size_t pos = path.find_last_of('/');
  if (pos != std::string::npos) {
    return path.substr(pos + 1);
  }
  pos = path.find_last_of('\\');
  if (pos != std::string::npos) {
    return path.substr(pos + 1);
  }
  return path;
}

const char* rng_type_to_str[] = {
  "std_default",
  "cuda",
};

const char* sample_method_str[] = {
  "euler_a",
  "euler",
  "heun",
  "dpm2",
  "dpm++2s_a",
  "dpm++2m",
  "dpm++2mv2",
  "lcm",
};

// Define a map to convert strings to sample_method_t enum values
std::map<std::string, sample_method_t> sample_method_map = {
  {"euler_a", EULER_A},
  {"euler", EULER},
  {"heun", HEUN},
  {"dpm2", DPM2},
  {"dpm++2s_a", DPMPP2S_A},
  {"dpm++2m", DPMPP2M},
  {"dpm++2mv2", DPMPP2Mv2},
  {"lcm", LCM}
};

static void parse_options_generation(const json &body, sd_params& params)
{
  sd_params default_params;
  params.prompt = json_value(body, "prompt", default_params.prompt);
  params.negative_prompt = json_value(body, "negative_prompt", default_params.negative_prompt);
  params.seed = json_value(body, "seed", default_params.seed);
  params.sample_steps = json_value(body, "steps", default_params.sample_steps);
  // Parse sample_method as a string
  std::string sample_method_str = json_value<std::string>(body, "sampler", "euler_a");
  // Convert the string to the corresponding enum value using the map
  if (sample_method_map.count(sample_method_str) > 0) {
    params.sample_method = sample_method_map[sample_method_str];
  } else {
    // Handle the case where the string does not match any enum value
    // Here we default to EULER_A, but you can handle this differently if you want
    params.sample_method = default_params.sample_method;
  }

  params.sample_method = json_value(body, "sampler", default_params.sample_method);
  params.batch_count = json_value(body, "batch_count", default_params.batch_count);
  params.width = json_value(body, "width", default_params.width);
  params.height = json_value(body, "height", default_params.height);
  params.cfg_scale = json_value(body, "cfg_scale", default_params.cfg_scale);
}

static void parse_options_model(const json &body, sd_params& params) {
  sd_params default_params;
  params.model_path = json_value(body, "model_path", default_params.model_path);
  params.vae_path = json_value(body, "vae_path", default_params.vae_path);
  params.lora_model_dir = json_value(body, "lora_model_dir", default_params.lora_model_dir);
  params.n_threads = json_value(body, "n_threads", default_params.n_threads);
  params.rng_type = json_value(body, "rng_type", default_params.rng_type);
  // tae is default required, add it in the docker image
  params.taesd_path = json_value(body, "taesd_path", default_params.taesd_path);
  // make wtype as empty
  params.wtype = GGML_TYPE_COUNT;
}


std::string base64_encode(const uint8_t* buf, unsigned int bufLen) {
  std::string base64;
  int i = 0;
  int j = 0;
  uint8_t char_array_3[3];
  uint8_t char_array_4[4];

  while (bufLen--) {
    char_array_3[i++] = *(buf++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for(i = 0; (i <4) ; i++)
        base64 += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i)
  {
    for(j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
    char_array_4[3] = char_array_3[2] & 0x3f;

    for (j = 0; (j < i + 1); j++)
      base64 += base64_chars[char_array_4[j]];

    while((i++ < 3))
      base64 += '=';
  }

  return base64;
}

void sdCPP::Text2Img(
    const HttpRequestPtr& req,
    std::function<void(const HttpResponsePtr&)>&& callback) {
  if(!sd_model_is_loaded()) {
    // return error
    Json::Value jsonResp;
    jsonResp["message"] = "Model is not loaded";
    auto resp = nitro_utils::nitroHttpJsonResponse(jsonResp);
    resp->setStatusCode(drogon::k500InternalServerError);
    callback(resp);
    return;
  }
  sd_params params;
  // parse request to sd_params
  auto jsonReq = req->getJsonObject();
  auto inputObject = (*jsonReq)["input"];
  if (inputObject.empty()) {
    // return error
    Json::Value jsonResp;
    jsonResp["message"] = "Input is empty";
    auto resp = nitro_utils::nitroHttpJsonResponse(jsonResp);
    resp->setStatusCode(drogon::k400BadRequest);
    callback(resp);
    return;
  }
  parse_options_generation(json::parse(inputObject.toStyledString()), params);
  LOG_INFO << "txt2img start";
  sd_image_t* images = txt2img(sd,
                            params.prompt.c_str(),
                            params.negative_prompt.c_str(), 0,
                            params.cfg_scale, params.width, params.height,
                            params.sample_method, params.sample_steps, params.seed,
                            params.batch_count, NULL, 0.0f, 0.0f, 0.0f, "");

  LOG_INFO << "txt2img end";
  Json::Value base64_images;
  for(int i = 0; i < params.batch_count; i ++) {
    int len;
    sd_image_t image = images[i];
    uint8_t* png = stbi_write_png_to_mem((const unsigned char *) image.data, 0, params.width, params.height, 3, &len);
    // add base64 png prefix
    std::string base64_png = "data:image/png;base64," + base64_encode(png, len);
    base64_images.append(base64_png);
  }
  LOG_INFO << "base64_images ends";
  Json::Value jsonResp;
  jsonResp["output"] = base64_images;
  auto resp = nitro_utils::nitroHttpJsonResponse(jsonResp);
  callback(resp);
}

void sdCPP::LoadModel(
    const HttpRequestPtr& req,
    std::function<void(const HttpResponsePtr&)>&& callback) {
  if (!nitro_utils::isAVX2Supported() && ggml_cpu_has_avx2()) {
    LOG_ERROR << "AVX2 is not supported by your processor";
    Json::Value jsonResp;
    jsonResp["message"] = "AVX2 is not supported by your processor, please download and replace the correct Nitro asset version";
    auto resp = nitro_utils::nitroHttpJsonResponse(jsonResp);
    resp->setStatusCode(drogon::k500InternalServerError);
    callback(resp);
    return;
  }

  sd_params params;

  // init params from request
  auto jsonReq = req->getJsonObject();
  if (jsonReq) {
    // convert Json::value to nlohmann::json;
    json body = json::parse(jsonReq->toStyledString());
    parse_options_model(body, params);
  }
  sd = new_sd_ctx_direct(params.lora_model_dir.c_str(), true, false, params.n_threads, params.rng_type);
  if(!sd_model_is_loaded()) {
    load_model(sd,
        params.model_path.c_str(),
        params.vae_path.c_str(),
        params.taesd_path.c_str(),
        "", "", (sd_type_t)params.wtype, false, params.schedule, false);
    set_model_loaded();
    LOG_INFO << "sd model loaded successfully";
  }
  // return success
  Json::Value jsonResp;
  jsonResp["message"] = "Model loaded successfully";
  auto resp = nitro_utils::nitroHttpJsonResponse(jsonResp);
  callback(resp);
}


void sdCPP::ModelStatus(
    const HttpRequestPtr& req,
    std::function<void(const HttpResponsePtr&)>&& callback) {
  Json::Value jsonResp;
  bool is_model_loaded = sd_model_is_loaded();
  jsonResp["model_loaded"] = is_model_loaded;
  jsonResp["model_data"] = "dummy";
  auto resp = nitro_utils::nitroHttpJsonResponse(jsonResp);
  callback(resp);
}






