from openvino.inference_engine import IENetwork, IEPlugin

class NetworkLoader:
  @staticmethod
  def load (modelTopo, modelWeights):
    try:
      net = IENetwork(modelTopo, modelWeights)
      plugin = IEPlugin(device='MYRIAD')
      excNet = plugin.load(network=net, num_requests=2)
      return net, excNet
    except Exception as e:
      raise e
    