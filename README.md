Download link :https://programming.engineering/product/from-3d-reconstruction-to-recognition-homework-3/


# From-3D-Reconstruction-to-Recognition-Homework-3
From 3D Reconstruction to Recognition Homework #3
Overview


Figure 1: Our nal carving using the true silhouette values

our voxel grid. Implement this carving process in carve(). Submit your code and a picture of what it looks like after one iteration of the carving. [15 points]

The last step in the pipeline is to carve out multiple views. Submit the nal output after all carvings have been completed, using num voxels 6; 000; 000. [5 points]

Notice that the reconstruction is not really that exceptional. This is because a lot of space is wasted when we set the initial bounds of where we carve. Currently, we initialize the bounds of the voxel grid to be the locations of the cameras. However, we can do better than this by completing a quick carve on a much lower resolution voxel grid (we use num voxels = 4000) to estimate how big the object is and retrieve tighter bounds. Complete the method get voxel bounds() and change the variable estimate better bounds in the main() func-tion to True. Submit your new carving, which should be more detailed, and your code. [10 points]

Finally, let’s have a fun experiment. Notice that in the rst three steps, we used perfect silhouettes to carve our object. Look at the estimate silhouette() function implemented and its output. Notice that this simple method does not produce a really accurate silhouette. However, when we run space carving on it, the result still looks decent!

Why is this the case? [2 points]

What happens if you reduce the number of views? [1 points]

What if the estimated silhouettes weren’t conservative, meaning that one or a few views had parts of the object missing? [2 points]

Representation Learning (10 points)

In this problem, you’ll implement a method for image representation learning for clasifying clothing items from the Fashion MNIST dataset.


Figure 2: The Fashion MNIST dataset

Unlike the previous problems, for this one you will have to work with Google Colaboratory. In Google Drive, follow these steps to make sure you have the ability to work on this problem:

Click the wheel in the top right corner and select Settings.

Click on the Manage Apps tab.

At the top, select Connect more apps which should bring up a GSuite Marketplace window.

Search for Colab then click Add.

Now, upload \p3/RepresentationLearning.ipynb\ and the contents inside ’p3/code’ to a location of your choosing on Drive. Then, navigate to this folder and open the le \RepresentationLearn-ing.ipynb\ with Colaboratory. The rest of the instructions are provided in that document. Note that there is no autograder for this problem, as you should be able to con rm whether your imple-mentation works from the images and plots. Include the following in your writeup:

a. Once you nish the section “Fashion MNIST Data Preparation”, include the 3 by 3 grid visualization of the Fashion MNIST data [1 point].

Once you nish the section “Training for Fashion MNIST Class Prediction”, include the two graphs of training progress over 10 epochs, as well as the test errors [4 points].

Once you nish the section “Representation Learning via Rotation Classi cation”, include the 3 by 3 grid visualization and training plot, as well as the test error [4 points].

Once you nish the section “Fine-Tuning for Fashion MNIST classi cation”, include all 3 sets of graphs from this section, as well as the test errors [1 point].

Fun fact: the method you just implemented if from the ICLR2018 paper “Unsupervised Rep-resentation Learning by Predicting Image Rotations”.

Monocular Depth Estimation (10 points)

Now that you’ve had some experience with representation learning on a small dataset, we’ll move on to the more complex task of training a larger model on the task of monocular depth estimation.

We will be implementing the approach taken in “High Quality Monocular Depth Estimation via Transfer Learning”, which showed that taking a big model pre-trained on classifying objects from



Figure 3: The task this problem will involve

the ImageNet dataset helps a lot with training that model to perform monocular depth estimation on the NYU Depth v2 dataset. We created a smaller and simpler dataset that is a variation on the CLEVR dataset for this problem that we call CLEVR-D, so we’ll actually not be using the pre-trained features, but otherwise the approach is the same.

As with the previous problem, you’ll be working with Colaboratory, so once again begin by uploading the contents of ’p3/code’ to a location of your choosing on Google Drive. Then, open the le \MonocularDepthEstimation.ipynb\ with Colaboratory. The rest of the instructions are provided in that document. Note that there is no autograder for this problem, as you should be able to con rm whether your implementation works from the images and plots. Include the following in your writeup:

a. Once you nish the section “Checking out the data”, include the grid visualization of the CLEVR-D data [5 points)].

Once you nish the section “Training the model”, include a screenshot of your train and test losses from Tensorboard as well as the nal outputs of the network [5 points].

Extra Credit As described at the bottom of the Colab notebook, you may optionally try to do representation learning by using an autoencoder, and see if that helps with training for monocular depth prediction. If you decide to do this, include a several sentence summary of how you went about it as well as sentence or two about the results, plus plots from Tensorboard and visualizations of the network in action [20 points].

Unsupervised monocular depth estimation (20 points)


Figure 4

In this problem, we will take a step further to train monocular depth estimation networks without ground-truth training data. Although neural networks best train with large-scale training data, it is often challenging to collect ground-truth data for every domain of problem. For example, Microsoft Kinect, one of the most popular depth camera, uses infrared camera that does not


work outdoors, and training monocular depth estimation networks for outdoor scenes can be more challenging.

We instead will utilize the knowledge we learned about stereo computer vision in this course to train monocular depth estimation networks without ground-truth data. In this problem, we will train a network to predict disparity. As shown in Figure 4, disparity(d) is simply inverse proportional to depth(z), which still serves our purpose. Given a pair of left and right view of recti ed images as inputs, we can synthesize right image by shifting left image toward right by the disparity and vice versa. We will utilize this trait to synthesize both left and right images and enforce them to look similar to the original left and right images.


Figure 5

Figure 5 is a summary of how unsupervised monocular depth estimation works. This method is derived from the paper “Unsupervised Monocular Depth Estimation with Left-Right Consistency”. The networks take left view of the stereo image imgl as input and outputs two disparity maps displ (the disparity map of the left view that maps the right image to the left) and dispr (the disparity map of the right view that maps the left image to the right). Although the network only takes left image as an input, we train the network to predict disparity of both left and right sides. This design allows us to make monocular depth prediction possible (i.e. does not take stereo images as input) and enforce cycle consistency between the left and right view of the stereo images.

Then, assuming that the input images are recti ed, we can generate left and right images from the predicted disparities. To be more concrete, using the left disparity displ, we synthe-size left image and disparity as following: imgl0 = generate image left(imgr; displ) and disp0l = generate image left(dispr; displ). Similarly, using the right disparity dispr, we synthesize right im-age and disparity as following: imgr0 = generate image right(imgl; dispr) and

disp0r = generate image right(displ; dispr). We will ask you to implement generate img left and generate image right in this pset.

In order to predict a reasonable disparity that can shift left image to right and vice versa, we com-pare the synthesized image with real image: Limg = comparei(imgl0 ; imgl) + comparei(imgr0 ; imgr). For completeness, comparei is L1 and SSIM.

In order to enforces cycle consistency between the left and right disparities, we compare the syn-thesized disparity with predicted disparity: Ldisp = compared(disp0l; displ) + compare(disp0r; dispr).


Figure 6: Input and output (left/right disparity) of trained monocular depth estimation networks.

For completeness, compared is L1.

Please ll in the codes at p4/problems.py as outlined below. For running the code, you’ll have to instal PyTorch and torchvision, and then you can test it out by running it locally ‘python problems.py‘. Alternatively, you can use the ipython notebook ‘problems.ipynb‘ with Google Colab as in the previous two problems { for this problem you can just upload the les directly to the

Optionally, if you want to train the network yourself, you may follow the instruction on the README le. We will not ask you to train the model since it is too computation-intensive.

Before we get started, we would like you to implement a data augmentation function for stereo images that randomly ips the given image horizontally. In neural networks, data augmentation takes a crucial role in better generalization of the problem. One of the most common data augmentation when using 2D images as input is to randomly ip the image horizontally. One interesting di erence in our problem setup is that we take a pair of recti ed stereo images as input. In order to maintain the stereo relationship after the horizontal ip, it requires a special attention. Please ll in the code to implement the data augmentation function. In your report include the images generated by this part of the code (no need to include the input images). [5 points]

Implement a function bilinear sampler which shifts the given horizontally given the disparity. The core idea of unsupervised monocular depth estimation is that we can generate left image from right and vice versa by sampling recti ed images horizontally using the disparity. We will ask you to implement a function that simply samples image with horizontal displacement as given by the input disparity. In your report include the images generated by this part of the code (no need to include the input images). [5 points]

Implement functions generate image right and generate image left which generates right view of the image from left image using the disparity and vice versa. This will be a simple one-liner that applies bilinear sampler. In your report include the images generated by this part of the code (no need to include the input images). [5 points)]

In Figure 6, we visualize output of the networks trained with the losses you have implemented. You may notice that there are some boundary artifacts on the left side of the left disparity and right side of the right disparity. Brie y explain why. [5 points]

Tracking with Optical and Scene ow (20 points)

Lastly, a problem having to do with optical and scene ow. Consider the scenario depicted in Fig. 7, where an object O is being interacting by a person, while the robot observes the interaction with its RGB-D camera C. Files rgb01.png, rgb02.png, …, rgb10.png in the globe1 folder contain ten RGB frames of the interaction as observed by the robot. Files depth01.txt, depth02.txt, …, depth10.txt contain the corresponding registered depth values for the same frames.


CF


p5 data/globe2/rgb02.png, p5 data/globe1/rgb04.png,

p5 data/globe1/rgb06.png, p5 data/chairs/frame chairs 1.png,

p5 data/chairs/frame chairs 2.png) to the notebook les tab. Run the cells in the note-book to produce the optical ow between the pairs of images, and include these ow images (and the reconstructed images) in your report. Note any qualitative di erences you see be-tween the optical ow results for the di erent pairs of images. Did one of the optical ow results and reconstructions from a speci c pair of the images look less noisier than the oth-ers? Describe why you think this might be the case, and if there are any ways in which the performance on the other pairs of images could be improved with the FlowNet model. [3 points]
