The goal of this project is to identify lymphoproliferative disorders —a category of cancer—
in individuals with lymphocytosis, marked by an elevated count of lymphocytes.
The primary challenge lies in discerning this condition, as lymphocytosis frequently
occurs in patients and may simply signal a benign response to various factors, such as
infections. Differentiating between benign and malignant causes of lymphocytosis remains
a significant challenge in clinical practice. Conventional methods, such as visual inspection
of blood smears, together with taking into account the patient’s clinical data, often lack
the necessary precision. Other methods like flow cytometry offer increased accuracy, but
their cost and limited availability hinder widespread adoption. This project seeks to address
these limitations by developing a new automated system to assist clinicians in identifying
patients who might require further evaluation through flow cytometry.
To achieve this goal, we used a dataset where the anonymized blood smears and the
corresponding demographic information were collected from 204 patients diagnosed with
elevated lymphocyte counts at Lyon Sud University Hospital. The dataset is segmented,
with 142 patients allocated for training the model and the remaining 42 reserved for testing.
Our proposed model takes a multifaceted approach, incorporating both image data from
blood smears and patient metadata. A pre-trained ResNet34 convolutional
neural network (CNN) will be employed to extract key features from the blood smear images.
To summarize the information contained in the sequence of images associated with a patient,
an attention mechanism will be used to aggregate the features of the images. Furthermore,
a separate branch within the model will process patient metadata. By combining these
elements and passing them through a classifier, we aim to develop a robust and informative
model.
In this project, we will not only develop the model architecture but also delve deeper to
optimize its performance. We will explore various strategies for hyperparameter selection,
validation methodology, preprocessing of images, and model configurations.
