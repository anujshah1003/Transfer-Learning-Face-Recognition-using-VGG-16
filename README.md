# Transfer-Learning-Face-Recognition-using-VGG-16
This repository shows how we can use transfer learning in keras with the example of training a face recognition model using VGG-16 pre-trained weights.The example shown here is based on Refik Can Malli keras-vggface: https://github.com/rcmalli/keras-vggface.
The vggface is the famous VGG-16 CNN trained on 2.6 million images of 2,622 different identities. the paper Deep Face Recognition describes it well : http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf

vgg face descriptor: http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

vggface data : http://www.robots.ox.ac.uk/~vgg/data/vgg_face/

Transfer learning refers to the technique of using knowledge of one domain to another domain.i.e. a NN model trained on one dataset can be used for other dataset by fine-tuning the former network.

Definition : Given a source domain Ds and a learning task Ts, a target domain Dt and learning task Tt, transfer learning aims to help improve the learning of the the target predictive function Ft(.) in Dt using the knowledge in Ds and Ts, where Ds ≠ Dt, or Ts ≠ Tt.

A good explanation of how to use transfer learning practically is explained in http://cs231n.github.io/transfer-learning/

The vggface model weights is loaded as such without including the last layers by calling

	VGGFace(include_top=False, weights='vggface',input_tensor=None) 

from keras-vggface : https://github.com/rcmalli/keras-vggface.
          
only the last three dense layers are fine tuned as per our requirement. All the layers of the vggface network are made non-trainable except the last three layers  by using 

	layer_count = 0
	for layer in custom_vgg_model.layers:
		layer_count = layer_count+1
	for l in range(layer_count-3):
		custom_vgg_model.layers[l].trainable=False
  
 
