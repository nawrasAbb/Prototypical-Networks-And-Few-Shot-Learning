# Prototypical networks and few-shot learning
This project was the final part of the Deep Learning course. We were a team of 2 partners.

# Project goal(s)
Implement few short learning procedures using prototypical networks approach.
The main goal of the project is to implement prototypical network with CNN (based on [1]) and then try to improve the results without using a significantly more complex architecture.
Steps we have planned:
1.	Create CNN for prototypical network following the description in [1]
2.	Reproduce the results from the paper [1] – use it as a baseline
3.	Tweak the CNN configuration to improve performance
4.	Implement and use other possible networks
  a.	Densely connected CNN
  b.	Residual Network
  c.	Residual Network with Stochastic Depth

# Models architecture
In our project we used 4 different model of various sizes and types. 
(Please refer to appendix 3 to see models’ summaries).
1.	Convolutional network with constant number of feature maps per block

![image](https://user-images.githubusercontent.com/97045152/148187450-14cce10b-6e72-4704-931c-928fac402a14.png)
This architecture closely follows the setup from the original paper [1]



2.Convolution network with dense layers
This architecture is similar to previous one, but we have added residual connections and increased the number of convolutions. A very close architecture was used in [2].

![image](https://user-images.githubusercontent.com/97045152/148187698-49f527ae-5461-48c0-8bf0-cc34ee464661.png)

Setup used:
  - Convolution 3x3 with 64 masks was used
  - Max pooling was 2x2
 
3.Residual deep network based on ResNet18 
We have implemented the architecture on network identical to described in [3].

![image](https://user-images.githubusercontent.com/97045152/148188920-8a850402-009d-4ad5-91ec-0a379b7b71dc.png)

Where:

•	Basic block is:

![image](https://user-images.githubusercontent.com/97045152/148189002-8edb7e02-cbe7-4f58-a105-d1e0ae667a79.png)


•	Down sample block is the same as basic but with down sample stride convolution before activation function

![image](https://user-images.githubusercontent.com/97045152/148189056-ae206f30-a7d2-436c-b22b-4f432522a689.png)

4.	Residual network with Drop Blocks
We took the implementation of this residual network from 
https://github.com/corwinliu9669/Learning-a-Few-shot-Embedding-Model-with-Contrastive-Learning/blob/main/resnet.py 
(presented in article [9] and used in the paper [4]) and slightly simplified it - mainly by removing the “average pooling” part. 

![image](https://user-images.githubusercontent.com/97045152/148189587-0bfcb017-f306-469b-9752-0f7e7819c716.png)


![image](https://user-images.githubusercontent.com/97045152/148189664-1b253a4e-7e2a-43a8-8468-f137199ae27a.png)


# Data & Processing 
- Dataset
We used a very well-known mini-ImageNet dataset that contains pictures of size 84x84, joined into 100 classes (from [5])
https://www.kaggle.com/whitemoon/miniimagenet 

- Splits
100 classes are divided into 64, 16, and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test, as proposed in [6]
From Kaggle we have downloaded Python pickle files that already divided into test, validation and train datasets.
![image](https://user-images.githubusercontent.com/97045152/148189937-a8c21b3d-afcb-4ff0-aa8d-d1f92b08bc12.png)
No further preprocessing was made to images.


# Data Loading 
1.	Manual loading from files:
We used Pickle to load the data files. Each data file is loaded into dictionary with 2 keys
![image](https://user-images.githubusercontent.com/97045152/148191172-87f6c143-907a-4893-ab58-e4c72dda0e2b.png)


2.	Episodes creation:
Each training episode contains of n-ways and k-shot, where “# ways” is the number of classes per episode, and “# shots” is the number of shots per class.
We have implemented a simple algorithm that randomly chooses defined number of ways and shots from data dictionary and thus creates an episode.


# Optimizations
1.	Adam optimizer was our default choice, since it has shown the best speed and performance during learning. We also used the default settings for it.
2.	Learning rate: 
We used Step LR scheduler to reduce learning rate by 50% every defined number of steps.


# Experiments
Training & Evaluation process
We have followed the standard training and evaluation procedure:
•	100 episodes per epoch 
•	10K max epochs (in fact we have never got even to 1000 epochs)
•	early stopping with patience 100 initially, later we have changed it to 200
•	loss function was implemented according to [1] and all models use the same loss calculation
•	learning rate started from 0.002 and reduced each 20 epochs 
(later changed the steps value to 50)
•	evaluation was made at 5-ways-5-shots episodes, while in training number of shots was different (under the restriction of GPU memory capacity)



# Model 1 – CNN
1.	Our first milestone was reproducing the results from paper [1]. 
  We have implemented the model 1 as CNN and tried a different number of ways in training:
  
  ![image](https://user-images.githubusercontent.com/97045152/148192298-b2afe50a-373e-4f15-a9f5-7a2236766101.png)
  
  According to the researches, best performance was achieved when training with more ways, than in evaluation. In the paper they have got to 68.2, while using 20 ways per episode. 
  Unfortunately, we have only 1 GPU (and not a very powerful one), so the maximum ways we have got is only 16. 
  From the paper [1], for 5-way-5-shot:
  ![image](https://user-images.githubusercontent.com/97045152/148192366-e3eca90a-9e82-4cef-97fd-2241e7883c90.png)
  In any case from now we consider 66.90% accuracy as our baseline and we try to improve it.
  
2.	Adding the dropout.
Since overfitting is the worst problem we face in few-shot learning, we have tried to increase generalization by adding some dropout to the model. 

![image](https://user-images.githubusercontent.com/97045152/148192951-b82f2ed1-40f9-4bbd-9ce2-958b7ed51a86.png)

Clearly, it didn’t work. 
Not only we didn’t improve the score, but made it significantly worse. Looks like dropout was completely messing the score.


3.	Mahalanobis distance vs. the Euclidean one.
Next idea was to try some other metric and we have chosen the Mahalanobis distance, which is a generalization over the Euclidian distance. We have implemented the calculations part and run the training, but it didn’t perform well.
The calculation required to work with matrices of dimension 1600x1600 (according to the embedding size for our model), and even on CUDA we had severe raise in runtime.

![image](https://user-images.githubusercontent.com/97045152/148193033-32ffbdec-e6e4-4f0a-9d2f-151d9d929765.png)

Since Mahalanobis distance was 25 times slower than the basic one, it was impossible to run for such a long time, and we have decided to try other methods.

Note: we have tried to add some FC layers after the CNN to reduce output dimension and run calculations faster, but this version of network just stopped learning (output vector was not large enough to support proper learning).

4.	Adding temperature parameter to distance calculation. 
Authors of paper [2] suggested to use metric scaling to improve the accuracy of few-shot learning. Instead of just calculating the Euclidian distance we have added a learnable parameter (alpha) to be used as scaling factor in loss calculation 

![image](https://user-images.githubusercontent.com/97045152/148193139-097a82a7-3461-4264-b22b-ff342c2bf33d.png)

We indeed had the improvement of approx. 0.5% (not much, unfortunately), but we have decided to use the scaling for further steps.
5.	Increasing number of feature maps.
Now we have tried to use more complex models (under the same architecture) by increasing the number of convolution maps in each stage. Due to hardware limitations we had to decrease number of ways for training to still be able to run without “out-of-GPU-memory” error.

![image](https://user-images.githubusercontent.com/97045152/148193217-af8c4e7a-be38-4f73-9852-11abc8e27b51.png)

It can clearly be seen that adding more maps allowed us to get better performance.
Also we checked the influence of metric scaling on a model with more maps and saw even a better improvement.
Note: model with 128 feature maps and 16 ways was trained in Google Collab to avoid failure with GPU errors.

6.	Fine tuning the learning rate
And last, but not the least we have tried several learning rates and scheduler step values to tune the accuracy to the maximum 
•	on a model with 128 maps and 8 way per training episode
•	each time the LR decreased by 50%

![image](https://user-images.githubusercontent.com/97045152/148193304-90a4163e-5002-4b87-9d39-8379e012ba3d.png)

After all, we have increased the accuracy percentage and found the maximum performance improvement on this model vs. baseline (68.36% vs. 66.9%) just by adding more masks and tuning the learning rate (plus some tiny minor metric scaling).

Note:  improved model has more parameters and thus training time has increased as well

![image](https://user-images.githubusercontent.com/97045152/148193368-d7d4af89-48f3-4b2f-8bb8-623c71c47c40.png)

# Model 2 – Dense layers CNN
After getting the improvement with regular CNN model we have decided to add some residual connections to create a “dense” model.
Unfortunately, the performance was awful – only around 50%.

This model was an unsuccessful try, but still we wanted to continue with other residual networks.

# Model 3 – Residual network
For this model we took the classical ResNet18 architecture (from paper [3]).

![image](https://user-images.githubusercontent.com/97045152/148193494-c12afbd8-3808-4659-9337-a16ae542f2c7.png)

Once again, we have tried to add a dropout (after ReLU), but still we saw the same familiar effect of reducing the performance. On the other hand, the accuracy without the dropout was around 60%, that was promising.
Here the number of parameters was huge (11,176,512) – and we thought that it might be a good idea to look for the smaller (much smaller) residual network.

# Model 4 – Residual network with drop blocks
We have found architecture (ResNet12) used in [8], following the practice of papers [2] and [7]. 
To save time, we simply took the ready implementation from the GitHub https://github.com/kjunelee/MetaOptNet 
removed the last averaging layer and tried various tweaks.
(We could run the network only with 5 ways per train episode)

1.	Changing the number of feature maps.
In original paper, the researchers have used [64, 160, 320, 640] maps configuration. We have tried several other options, and only with 5-ways per train episode, since all other values failed to run under GPU restrictions.

![image](https://user-images.githubusercontent.com/97045152/148196901-8cf4f7ec-ea4c-49b6-a22e-9d7a41f1472d.png)

Surprisingly the configuration with 64 maps performed the best (similar to CNN model 1 with the same number of maps, with 66.9%). 
Number of parameters for [64, 64, 64, 64] maps is 421,760.

2.	Different drop block rate.
This architecture allows to remove some areas from feature maps (with certain probability) during training. It was introduced in [9] and shown that it outperforms other techniques like regular dropout, cutout and others.
Default drop block rate was set to 0.1, but we have tried also 0.5 (using the [64, 64, 64, 64] maps setup).

![image](https://user-images.githubusercontent.com/97045152/148196942-693e2eef-d2eb-4f58-b084-ed3fff6baeb5.png)

Higher drop rate reduced the accuracy (as we predicted, but the score was lower than we expected).
We have checked the root cause of low performance and that led us straight to the main improvement.

3.	Removing nodes dropout completely.
One of the features of this architecture was applying regular dropout on neurons if block dropout was not applied. 
But for all previous experiments we have clearly seen that nodes dropout dramatically reduces the accuracy, so we decided to remove it – and it raised the score significantly!

![image](https://user-images.githubusercontent.com/97045152/148196990-b59d89fd-149d-498e-9a67-ad9b3e54f6ff.png)

Interesting to mention, that in [9] authors also got the best performance on ImageNet with drop rate 0.1 (or keep probability 0.9 which is the same as drop 0.1).

![image](https://user-images.githubusercontent.com/97045152/148197050-17ef8c89-4e97-4757-b6f3-b0e6f9b9f00a.png)

# Results
According to our experiments, best result is achieved with:
•	The 4th Model (Residual network with drop blocks)
o	Using temperature distance parameter
o	And with some dropping of network residual blocks during training
It is worth to mention that the best Model 4 (residual), uses less parameters (421K vs. 447K) than the best Model 1 (CNN with 128 maps), and achieves better performance (70.9% vs. 68.36%), while consuming less GPU resources as well.

# Models summaries

# Model 1 – CNN:

![image](https://user-images.githubusercontent.com/97045152/148197787-f0a90b95-19b1-4b32-b1f7-571bfbbe0543.png)


# Model 2 – Dense layers CNN  

![image](https://user-images.githubusercontent.com/97045152/148197879-f9242efa-60c0-46d8-85dc-e6682db48d53.png)


# Model 3 – Residual network based on ResNet18

![image](https://user-images.githubusercontent.com/97045152/148197995-d14c9c98-2e54-4a2c-abd3-6101f40203ef.png)

![image](https://user-images.githubusercontent.com/97045152/148198062-78cc3980-72b1-438a-bcf4-e86c94ac9feb.png)


# Model 4 – Residual network based on resnet12

![image](https://user-images.githubusercontent.com/97045152/148198896-3bfcbbc1-ecc6-49b9-a43d-67c22a992df7.png)


# Papers and material used
1.	“Prototypical Networks for Few-shot Learning” - https://arxiv.org/abs/1703.05175
2.	“TADAM: Task dependent adaptive metric for improved few-shot learning” - https://arxiv.org/abs/1805.10123v4 
3.	“Deep Residual Learning for Image Recognition” - https://arxiv.org/abs/1512.03385 
4.	“Learning a Few-shot Embedding Model with Contrastive Learning” - https://ojs.aaai.org/index.php/AAAI/article/view/17047 
5.	“Matching Networks for One Shot Learning” - https://arxiv.org/abs/1606.04080 
6.	“Optimization as a model for few-shot learning” - https://openreview.net/pdf?id=rJY0-Kcll 
7.	“A Simple Neural Attentive Meta-Learner” - https://arxiv.org/abs/1707.03141v3 
8.	“Meta-Learning with Differentiable Convex Optimization” - https://arxiv.org/abs/1904.03758v2 
9.	“DropBlock: A regularization method for convolutional networks” - https://arxiv.org/abs/1810.12890


