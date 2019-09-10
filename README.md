# InnerEye-classifiers
Color of daily objects says a lot about them. However, upon applying image filters, which are often available on social media, objects in an image undergo a color transformation. The objects in such an image often have color that confuses us or conveys other meaning. Therefore, applying an image filter is a kind of image editing.


<p align="center">
  <img width="128" height="192" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/sample_images/Me.jpg">
  <br>
  Figure: An example of unedited image.
</p>


<p align="center">
  <img width="128" height="192" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/sample_images/Me_nashville.jpg">
  <br>
  Figure: Edited image due to application of Nashville filter.
</p>


<p align="center">
  <img width="128" height="192" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/sample_images/Me_xpro2.jpg">
  <br>
  Figure: Edited image due to application of XPro2 filter.
</p>


This classifier, based on the survey result of the ongoing [InnerEye](http://inner-eye.herokuapp.com/) project, can distinguish such edited and unedited images. InnerEye project aims at understanding the credibility of social interaction in the presence of both edited and unedited images on the social media platforms.


In this classifier, the challenge is to recognize how the color style of an image is different from it's unedited (images on which no filters has been applied) counterpart. However, under different illumination condition this color style changes. Assuming there is an invariant relationship among the colors for each of the image filters, this classifier has been built. To compile the dataset required, images are sampled from the [Google Landmarks dataset](https://www.kaggle.com/google/google-landmarks-dataset) first and then different image filters are applied to the images. Both the unedited and the edited counterparts of those are present in the dataset.


Because the color style of an image has to be understood, the content and the color of an image has to be separated. We need to work with only the style of the input image. The style and content separating mechanism in the classifier is based on the autoencoder architecture in [MUNIT](http://openaccess.thecvf.com/content_ECCV_2018/html/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.html) InnerEye classifier is then cotrained on the images (image reconstruction) and the edited image labels (unedited or edited). The classifier is multi-targeted to improve accuracy- therefore, the class target labels are {'unedited', '_1977', 'aden', 'brannan', 'brooklyn', 'clarendon', 'earlybird', 'gingham', 'hudson', 'inkwell', 'kelvin', 'lark', 'lofi', 'maven', 'mayfair', 'moon', 'nashville', 'perpetua', 'reyes', 'rise', 'slumber', 'stinson', 'toaster', 'valencia', 'walden', 'willow', 'xpro2'}.


<p align="center">
  <img width="326" height="422" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/architectures/analytical_classifier.jpg">
  <br>
  Figure: Architecture of the classifier.
</p>


A classifier that does not separate style and content and classifies on the style overfits very quickly.

<p align="center">
  <img width="326" height="422" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/Sequential/History.png">
  <br>
  Figure: Accuracy of the sequential classifier.
</p>


A classifier that separates style and content and classifies on the style does not over fits itself and keeps improving the accuracy over a large number of iterations.

<p align="center">
  <img width="326" height="422" src="https://github.com/greenboal/InnerEye-classifiers/blob/master/analytical_classifier/History.png">
  <br>
  Figure: Accuracy of the analytical classifier.
</p>


The author of the classifier is available at 0905107.saab@ugrad.cse.buet.ac.bd
