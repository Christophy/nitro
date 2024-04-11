#pragma once

#include <common/base.h>
#include <drogon/HttpController.h>
#include <utils/json.hpp>




using json = nlohmann::json;

using namespace drogon;

namespace inferences {

class sdCPP : public drogon::HttpController<sdCPP>{
public:
  METHOD_LIST_BEGIN
  // list path definitions here
  METHOD_ADD(sdCPP::LoadModel, "/loadmodel", Post);
  METHOD_ADD(sdCPP::ModelStatus, "/modelstatus", Get);
  METHOD_ADD(sdCPP::Text2Img, "/text2img", Post);
  METHOD_LIST_END

  void Text2Img(
      const HttpRequestPtr& req,
      std::function<void(const HttpResponsePtr&)>&& callback);
  void LoadModel(const HttpRequestPtr& req,
                 std::function<void(const HttpResponsePtr&)>&& callback);
  void ModelStatus(
      const HttpRequestPtr& req,
      std::function<void(const HttpResponsePtr&)>&& callback) ;

private:
  std::string prompt;
  int width = 512;
  int height = 512;
  int steps = 4;
  std::string sampler_method;

};

}  // namespace inferences