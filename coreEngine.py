import abc, os
import numpy as np
import onnxruntime


class EngineBase(abc.ABC):
	'''
    Currently only supports Onnx/TensorRT framework
	'''
	def __init__(self, model_path):
		if not os.path.isfile(model_path):
			raise Exception("The model path [%s] can't not found!" % model_path)
		assert model_path.endswith(('.onnx', '.trt')), 'Onnx/TensorRT Parameters must be a .onnx/.trt file.'
		self._framework_type = None

	@property
	def framework_type(self):
		if (self._framework_type == None):
			raise Exception("Framework type can't be None")
		return self._framework_type
	
	@framework_type.setter
	def framework_type(self, value):
		if ( not isinstance(value, str)):
			raise Exception("Framework type need be str")
		self._framework_type = value
	
	@abc.abstractmethod
	def get_engine_input_shape(self):
		return NotImplemented
	
	@abc.abstractmethod
	def get_engine_output_shape(self):
		return NotImplemented
	
	@abc.abstractmethod
	def engine_inference(self):
		return NotImplemented


class OnnxEngine(EngineBase):

	def __init__(self, onnx_file_path):
		EngineBase.__init__(self, onnx_file_path)
		if (onnxruntime.get_device() == 'GPU') :
			self.session = onnxruntime.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider'])
		else :
			#try:
			self.session = onnxruntime.InferenceSession(onnx_file_path, providers=['DmlExecutionProvider'], sess_options=onnxruntime.SessionOptions())
			self.session.set_providers(['DmlExecutionProvider'], [{'device_id': 0}])
				#print("dmlexec")
			#except:
				#self.session = onnxruntime.InferenceSession(onnx_file_path)
				#print("nodml")
		self.providers = self.session.get_providers()
		self.engine_dtype = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
		self.framework_type = "onnx"
		self.__load_engine_interface()

	def __load_engine_interface(self):
		self.__input_shape = [input.shape for input in self.session.get_inputs()]
		self.__input_names = [input.name for input in self.session.get_inputs()]
		self.__output_shape = [output.shape for output in self.session.get_outputs()]
		self.__output_names = [output.name for output in self.session.get_outputs()]

	def get_engine_input_shape(self):
		return self.__input_shape[0]

	def get_engine_output_shape(self):
		return self.__output_shape, self.__output_names
	
	def engine_inference(self, input_tensor):
		output = self.session.run(self.__output_names, {self.__input_names[0]: input_tensor})
		return output
