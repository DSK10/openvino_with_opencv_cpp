#include "model.h"

void DSK::TEST::check(){
    std::cout << "Inside function";
}

std::vector<float> DSK::TEST::tmain_old(cv::Mat MatImage) {
        // -------- Get OpenVINO runtime version --------
        // slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------


        // const std::string args = TSTRING2STRING(argv[0]);
        const std::string model_path = "../models/face-reidentification-retail-0095.xml";
        const std::string image_path = "dog.bmp";
        const std::string device_name = "CPU";

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read a model --------
        // slog::info << "Loading model files: " << model_path << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        // printInputAndOutputsInfo(*model);

        OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
        OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

        // -------- Step 3. Set up input

        // Read input image to a tensor and set it to an infer request
        // without resize and layout conversions
    
        ov::element::Type input_type = ov::element::u8;
        // cv::Mat MatImage = cv::imread(image_path_arg);
        int width = int(MatImage.rows);
        int height = int(MatImage.cols);
        ov::Shape input_shape = {1, width, height, 3};


        // ov::Shape input_shape = {1, reader->height(), reader->width(), 3};
        // std::shared_ptr<unsigned char> input_data = reader->getData();
        // std::cout << typeid(MatImage.get()).name() <<  "TYPE "<<std::endl;
        // // std::cout << input_shape <<  "= IS "<<std::endl;
        // just wrap image data by ov::Tensor without allocating of new memory
        ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, MatImage.data);

        const ov::Layout tensor_layout{"NHWC"};

        // -------- Step 4. Configure preprocessing --------

        ov::preprocess::PrePostProcessor ppp(model);

        // 1) Set input tensor information:
        // - input() provides information about a single model input
        // - reuse precision and shape from already available `input_tensor`
        // - layout of data is 'NHWC'
        ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
        // 2) Adding explicit preprocessing steps:
        // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
        // - apply linear resize from tensor spatial dims to model spatial dims
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        // 4) Here we suppose model has 'NCHW' layout for input
        ppp.input().model().set_layout("NCHW");
        // 5) Set output tensor information:
        // - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(ov::element::f32);

        // 6) Apply preprocessing modifying the original 'model'
        model = ppp.build();

        // -------- Step 5. Loading a model to the device --------
        ov::CompiledModel compiled_model = core.compile_model(model, device_name);

        // -------- Step 6. Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        // -----------------------------------------------------------------------------------------------------

        // -------- Step 7. Prepare input --------
        infer_request.set_input_tensor(input_tensor);

        // -------- Step 8. Do inference synchronously --------
        infer_request.infer();

        // -------- Step 9. Process output
        const ov::Tensor& output_tensor = infer_request.get_output_tensor();


        // Print classification results
        
        ClassificationResult classification_result(output_tensor, {image_path});
        classification_result.show();
        // for (auto x: classification_result.getR()){
        //     std::cout << x;
        // }


        return classification_result.getR();
}
