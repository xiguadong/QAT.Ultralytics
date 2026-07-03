import onnx

input_path = "./yolo26_onnx/yolo26n.onnx"
output_path = "./onnx/yolo26n-backbone.onnx"

input_names = ["images"]
output_names = ["/model.23/Reshape_1_output_0","/model.23/Reshape_3_output_0","/model.23/Reshape_5_output_0",
                "/model.23/Reshape_output_0", "/model.23/Reshape_2_output_0", "/model.23/Reshape_4_output_0"]
onnx.utils.extract_model(input_path, output_path, input_names, output_names)


# input_path = "yolo26_onnx/qat_exp29_one2many_slim.onnx"
# output_path = "yolo26_onnx/qat_exp29_one2many_slim-cut.onnx"

# input_names = ["x"]
# output_names = ["dequantize_per_tensor_203","dequantize_per_tensor_237", "dequantize_per_tensor_283",
#                 "dequantize_per_tensor_291", "dequantize_per_tensor_297", "dequantize_per_tensor_303",
#                 "dequantize_per_tensor_314", "dequantize_per_tensor_324", "dequantize_per_tensor_334"]
# onnx.utils.extract_model(input_path, output_path, input_names, output_names)


# input_path = "yolo26_onnx/qat_exp32_one2many_slim_fisher.onnx"
# output_path = "yolo26_onnx/qat_exp32_one2many_slim_fisher-cut.onnx"

# input_names = ["x"]
# output_names = ["dequantize_per_tensor_203","dequantize_per_tensor_237", "dequantize_per_tensor_283",
#                 "fisher_float_node_DequantizeLinear_1826", "fisher_float_node_DequantizeLinear_1994", "fisher_float_node_DequantizeLinear_1872",
#                 "fisher_float_node_DequantizeLinear_2070", "fisher_float_node_DequantizeLinear_1918", "fisher_float_node_DequantizeLinear_2146"]
# onnx.utils.extract_model(input_path, output_path, input_names, output_names)
