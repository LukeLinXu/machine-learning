# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Proposal and Capstone Project

**Note**

The Capstone is a two-staged project. The first is the proposal component, where you can receive valuable feedback about your project idea, design, and proposed solution. This must be completed prior to your implementation and submitting for the capstone project. 

You can find the [capstone proposal rubric here](https://review.udacity.com/#!/rubrics/410/view), and the [capstone project rubric here](https://review.udacity.com/#!/rubrics/108/view). Please ensure that you are following directions correctly before submitting these two stages which encapsulate your capstone.

**In real project I use Keras instead of TFlearn mentioned in Proposal. The reason is:**

I compared the ImageAugmentation from TFlearn with ImageDataGenerator from Keras, it
turns out ImageDataGenerator has more powerful features than ImageAugmentation. This
class allows you to:
- configure random transformations and normalization operations to be done on your
image data during training
- instantiate generators of augmented image batches (and their labels) via .flow(data,
labels) or .flow_from_directory(directory). These generators can then be used with
the Keras model methods that accept data generators as inputs, fit_generator,
evaluate_generator and predict_generator.

So I picked Keras as out project main training tool.

Please email [machine-support@udacity.com](mailto:machine-support@udacity.com) if you have any questions.
