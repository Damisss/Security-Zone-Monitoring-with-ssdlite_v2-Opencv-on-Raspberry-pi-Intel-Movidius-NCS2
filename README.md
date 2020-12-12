# security_app_with_ssdlite_raspberry_pi_intel_neural_compute_stcik2
<img src="/result.gif" width="350" height="400"/>

This project demostrates how to use a pre-trained object detection model (ssdlite_mobilenet_v2) plus opencv to monitor a security zones for intruders on edge device (Raspberry pi + Intel Movidius Neural Compute Stick NCS2). it attempts to show how to deploy Intermediate Representation Graph of a pre-trained model from well known Tensorflow Object Detection API version 1 model zoo. Also, the project shows how to save CPU cycles by using a cascade of background subtraction and object detection. 

# Convert TensorFlow Object Detection to Openvino Intermediate representation

Please follow [Openvino](https://docs.openvinotoolkit.org/latest/index.html) to setup your env on a computer. After setting up your environment, download ssdlite_mobilenet_v2 from TFOD model zoo. From there, navigate to downloaded folder and then run below command provided by openvino team to generate Intermediate Representation Graph (model topography and weights files).

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py -m frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --data_type FP16 --generate_deprecated_IR_V7

# Setting up Raspberry pi

Please follow this [Pyimagesearch's](https://www.pyimagesearch.com/2018/02/12/getting-started-with-the-intel-movidius-neural-compute-stick) tutorial to setup the raspberry pi for Movidius NCS2

# Inference:

To run real time inference:
python detect_realtime.py --topo model/ssdlite_v2.xml --weights model/ssdlite_v2.bin

# References

  - https://www.pyimagesearch.com/raspberry-pi-for-computer-vision/
  - https://docs.openvinotoolkit.org/latest/index.html
  - https://www.pyimagesearch.com/2018/02/12/getting-started-with-the-intel-movidius-neural-compute-stick
