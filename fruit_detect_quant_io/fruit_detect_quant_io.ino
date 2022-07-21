#include "model.h" //Name of the Model !!!!!!!!!!!!!!!!!
#include "img.h" //image to test our model //fire image

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <string>



const int no_pixels = 6912;
const int no_classes =24;
const char* labels[no_classes]= {
"apple_6",         "apple_pink_lady_1",     "carrot_1",
"apple_braeburn_1",      "apple_red_1",     "cucumber_1",
"apple_crimson_snow_1",  "apple_red_2",     "cucumber_3",
"apple_golden_1",       "apple_red_3",     "eggplant_violet_1",
"apple_golden_2",        "apple_red_delicios_1",  "pear_1",
"apple_golden_3",        "apple_red_yellow_1",    "pear_3",
"apple_granny_smith_1",  "apple_rotten_1",      "zucchini_1",
"apple_hit_1",       "cabbage_white_1",     "zucchini_dark_1"};


// TensorFlow Lite for Microcontroller global variables
tflite::ErrorReporter *error_reporter = nullptr;//!!!!!!!!!!

const tflite::Model* tflu_model            = nullptr;
tflite::MicroInterpreter* tflu_interpreter = nullptr;
TfLiteTensor* tflu_i_tensor                = nullptr;
TfLiteTensor* tflu_o_tensor                = nullptr;
tflite::MicroErrorReporter tflu_error;


constexpr int tensor_arena_size = 100 * 1024;//hyperparameter!!!!!
byte tensor_arena[tensor_arena_size] __attribute__((aligned(16)));
float   tflu_i_scale      = 0.0f;
//float   tflu_o_scale      = 0.0f;
int32_t tflu_i_zero_point = 0;
//int32_t tflu_o_zero_point = 0;

inline int8_t quantize(float x, float scale, float zero_point)
{
  return (x / scale) + zero_point;
}

//inline float dequantize(int8_t x, float scale, float zero_point)
//{
//  return ((float)x - zero_point) * scale;
//}

/////////////////////////////////////////////////////////////////////
void tflu_initialization()// Model Initialization
{
  Serial.println("TFLu initialization - start");

  // Load the TFLITE model
  tflu_model = tflite::GetModel(TFLite_Models_model_tflite);//NAME OF THE MODEL BUT INSTEAD OF. USE _!!!!, path is also included in the name
  if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(tflu_model->version());
    Serial.println("");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println("");
    while(1);
  }

   tflite::AllOpsResolver tflu_ops_resolver;

  // Initialize the TFLu interpreter
  tflu_interpreter = new tflite::MicroInterpreter(tflu_model, tflu_ops_resolver, tensor_arena, tensor_arena_size, &tflu_error);

  // Allocate TFLu internal memory
  tflu_interpreter->AllocateTensors();

  // Get the pointers for the input and output tensors
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);

  // Get the quantization parameters (per-tensor quantization)

  const auto* i_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tflu_i_tensor->quantization.params);
  const auto* o_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tflu_o_tensor->quantization.params);

  tflu_i_scale      = i_quantization->scale->data[0];
  tflu_i_zero_point = i_quantization->zero_point->data[0];

  
   Serial.println("TFLu initialization - completed");
}




//////////////////////////////////////////////////////////////////////////

void setup() {
  Serial.begin(9600);
  while (!Serial);// wait for serial initialization


  tflu_initialization();
  delay(4000);
  Serial.println("Init is done");
}


////////////////////////////////////////////////////////

void loop() {
  unsigned long timeBegin = millis();
  
  
  // Initialize the input tensor
  for (int i = 0; i < no_pixels; i++) {
    tflu_i_tensor->data.int8[i] = quantize(inp_t[i]/255, tflu_i_scale, tflu_i_zero_point); ////////////////////////
  }


  // Run inference
  TfLiteStatus invoke_status = tflu_interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error invoking the TFLu interpreter");
    return;
  }
  

   int index_loc_highest_prob = -1;
   float highest_prob = -1.0;
   for (int i = 0; i < no_classes; i++){
    if (tflu_o_tensor->data.f[i]>highest_prob){
      highest_prob = tflu_o_tensor->data.f[i];
      index_loc_highest_prob = i;
    }
   }
   Serial.println("Predicted Class:");
   Serial.println(labels[index_loc_highest_prob]);
   Serial.println("with probability:");
   Serial.println(highest_prob);
  
  
  //execution time calculation
  unsigned long timeEnd = millis();
  unsigned long duration = timeEnd - timeBegin;
  double averageDuration = (double)duration / 1000.0;

  Serial.println("Inference Time");
  Serial.println(averageDuration);
  
  Serial.println();
  delay(4000);
}
